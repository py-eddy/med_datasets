CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?h?t?j~?   max       ?????+        ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M??>   max       PR$        ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ????   max       =?v?        ?   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @?
=p??
   max       @D޸Q??     
    ?   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??z?G?    max       @vw
=p??     
   *?   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @(         max       @N?           ?  4?   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @?L        max       @??`            5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ??o   max       >	7L        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A?J?   max       B%uf        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A?y   max       B%?z        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >? ?   max       C?uG        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >?c?   max       C?h?        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ]        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          3        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          +        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M??>   max       P9        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ??[W>?6z   max       ???	?        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ????   max       =?v?        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @?
=p??
   max       @D?
=p??     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??z?G?    max       @vw
=p??     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @$         max       @N            ?  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @?L        max       @???            U?   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @?   max         @?        V?   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ???X?e   max       ????vȴ        W?                     '                  (   #   +   $         !            D   ,   L         L   (      /         !               (                     $      &      %         X   ]      @            3   -         )Nm??N6N??N?<N??O?3hO?fN??O{{*O<?N??KN!??P??O??nO?-P??N}b?O??O??N?	=N?)?N!?PCJ?P9[?O?'gO2??NeIDPR$P"?O???O?g3N??N???OPO???N??O??O???O?C?Oj??O??Nw??OJ?xN?(?O!P??M??>Ow?"O??ON?FO^?#N??2PĚPWN?aYP=pN?<?N?e?N??O?}%O?>dN6g?N?]?O?x?????ě????
??o:?o:?o:?o:?o;o;ě?;?`B<t?<#?
<#?
<#?
<D??<D??<T??<e`B<e`B<e`B<u<u<u<?o<?o<?C?<?t?<?t?<???<??
<??
<?9X<?9X<???<???<?/<??h<?<???<???<???<???=o=+=C?=C?=t?=t?=??=#?
='??=,1=,1=0 ?=8Q?=<j=H?9=ix?=m?h=q??=?7L=?\)=?v?(($")5<BFB?5*)((((((????????????????????]gptw~?????????wtg]]?????????????????????????????????????????
.++
??????????????????????????????


???????????????????????????????????????????????????????????????????????????????????ddilq?????????????td???
#/>GNOH</#????SSWalz?????????znaUS 
#/<OX]cljaU</# ????????????????????+)-05BO[gszztogbNB5+?????????????????#(/<>?@@<6/+#|}??????????????||||??????????????????????????)/22-%???????? ELQMB)????/278AHN[gt???tog[B5/YRPT[_ht??????ythc[Y????????????????????bbkt??????????????lb??????????
$#
????)BTXVNB6)??????)+????????????????????????????
 #/16<=</#





????????????????????
#0>BJNMI<0#
????????????????????5,,6:BOR[\g[XSOJB@65! $/;HTallg`TOH;/)'!vqstwz?????????????v???????
 ???????)5BNTVVN)??429<HQTUH<4444444444???????? ? ???????????????  ??????????? 
#&-0.)#
?????(043)???????ghpt????thgggggggggg	#/<BIIHA*#
	????????????????????&/<HLUZ_^TH=<9/%??????

??????)5;65)???):BJR[_VUB6?QY_d^h????????tmoh[Q???????????????????????	&05)?????????????????????????????????????????????||????????????||||||????????	?????????????!#" ????xonz?????zxxxxxxxxxx???????

????????????????????????????ŭŹ??????????????ŻŹŵŭťŭŭŭŭŭŭ????????????????????????????????????????????????????????????????????????????????????????????????????????(?5?A?N?O?N?I?A?=?5?,?(????????(????)?5?B?N?Z?W?Q?N?B?)???????????????A?F?D?A?:?4?*?(???????߽ٽݽ???ûлܻ??????????????ݻܻлûûûû?ƧƳ?????????????????????ƳƮƧƟƚƚƧ?????'?4???G?@?<?3?'????????????????????	??????	??????????????????¿??????????¿º·½¿¿¿¿¿¿¿¿¿¿?y?????????Կ߿??ݿпĿ???????x?p?n?j?y???????????????m?`?G?;?)?(?+?6?F?`?m?y??E?E?E?FFFFF
E?E?E?E?E?E?E?E?E?E?E?Eپ????Ǿ̾ɾþ???????f?Z?U?V?F?N?[?f?????G?T?T?`?d?h?`?T?G?<?;?@?G?G?G?G?G?G?G?G?)?B?N?[?l?h?[?R?S?N?B?5?0?)?#?????)??????*?0?9?C?K?C?*???????????????????%?(?/?4?(???????????????????"?*?.?2?.?+?"??	????????	??????N?Z?g?r?g?]?Z?N?J?M?N?N?N?N?N?N?N?N?N?N?y???????????????????y?m?`?T?@?>?A?T?`?yƧ??????$???????ƳƚƁ?u?o?h?T?\?uƁƧ??)?6?B?[?h?q?y?y?t?h?[?O?1?)??????????????????????????????{?s?m?o?s?|?????????????????????????????????????????????????????????s?(???
???(?6?F?g??????????????"?(?'?"?	???????????????????????׾????	??	??????׾ʾ?????????????Óàì???????????????????÷ìÛÓÙÒÓ?#?/?<?H?O?U?I?H?<?;?/?)?#???"?#?#?#?#????(?)?)?(??????
???????ÇÓàìù??????ùìÓÍÇÄ?|?t?u?z?~Ç????????????????????f?^?X?K?H?S?Y?f?p???????(?.?(???? ??????????????????????????????ںֺϺֺ????H?T?a?h?m?l?i?h?a?T?H?;?(?"???!?/?;?H???)?6?B?O?]?`?]?[?O?B?9?)??????čĚĦĳ????????????????ĿĳĚčĄĀĀč?#?<?U?e?h?h?d?Z?W?J?<?0?#???
?? ??#?zÇÓÚßÓÇ?z?v?v?z?z?z?z?z?z?z?z?z?z?Ϲܹݹ????????????ܹϹù????????????ùϿ	?? ?"?)?"???	???????????????	?	?	?	????????????????????ּммּټ????	??"?1?;?=?9?/?"??	?????????????????	?????û̻Ȼû???????????????????????????D?D?D?D?D?EEEEEED?D?D?D?D?D?D?D?D??s???????????????????????????y?n?k?j?h?s???????????	??????????????????????n?z?}?z?t?u?v?r?n?a?U?H?<?0?#?&?/?<?H?n?Z?f?h?n?q?q?f?Z?X?O?P?V?Z?Z?Z?Z?Z?Z?Z?Z???4?C?E?=?4?)?'???????ܻл̻Ի̻Ӽ?'?4?M??????????r?M?4????????????'D{D?D?D?D?D?D?D?D?D?D?D?D?D?DD{DsDsD{D{???????????ɺ????????~?[?F?7?A?P?Y?r?????(?4?A?M?Z?Z?]?Z?M?A?4?(?$??????F?:?5?-?!???????????????!?-?:?F?F?4?@?M?M?Y?]?^?Y?M?F?@?4?4?1?4?4?4?4?4?4¿???????
??-?-??
????????????½¿¼¿???????????????????????????????y?z?~????Ź????????????ŹŷŷŹŹŹŹŹŹŹŹŹŹEuE?E?E?E?E?E?E?E?E?EuEqEoEjEuEuEuEuEuEu?ʼּ??????? ?????????ּʼǼ????ʼʼʼ? R I N = U 1 * Y M ? @ X < _ ' @ 4 7 - * - ` % [ @  : Z 1 1 { 3 a .   f 4 . / u 4 2 0 .   b Y O "  F S . b [ \ x O ] H & = * .  ?  ?    ;  A  o  ?  ?  "  ?    ;  z  h  ?  z  ?  ]  k    ?  .  1  ?  ?  o  |    ?  ?  G  ?  ?  ?  8  Z  0  +    L    ?  ?  ?  !  ?  ,  -  1  ?  ?  ?  P  ?    ?  /    ?  ?  ?  ?  ?  !??o??o<o;D??<t?<?/=#?
<e`B<??
<???<?t?<?C?=H?9=8Q?=T??=D??<?o=C?=@?<???<?C?<?t?=???=m?h=?j<?<ě?=??=ix?=,1=?+='??<???=aG?=L??<??h=8Q?=e`B=?O?=e`B=ix?=t?=D??=\)=8Q?=?O?=#?
=??P=e`B=??P=?+=@?>?>	7L=?O?=?;d=q??=?%=?o=?/=???=??P=?j>	7LB?B??B
$?BݮB?Br?B s?B$/BF?B!?BgbB2?BD?Bt?Bn?Ba?B??B|B?,B.*Bd?Bn?B?<B+?B?BsB=B@*B۩BJB??B?B?	B"?B%ufB?BU?A?J?B ??B??B?pB?Bq?B#GB$??B??B>?B??BJUB?B2Bs?B??B?1BxBg?Bp6B??BBg?B?A??BB=SB??BrB??B
H?B??BD]B:3B J?B$??B??B!`B?>BD2B|sB:?B:jBP?B?_B?B?cB?B?B>3BsBK?B?|BEB
? B?zBBCB??B?BC?B"@?B%?zB??B<?A?yB ??BB&B??B?BB?B#@?B%8?B?B??B?B?rB?ZBEzB??B?%B?B??B?YBEB??B9?B?*B??A???B?NB??A?"A??6A?v?A??A?.|A???A3@?? B???	?A?SA??As?BAi?$C?uGAF?=AfX?A??>A???A???A]#gA??iAn??B??A?>AHV?A???A?jOA??NASl?A???A??kA3SAˏ?@??nA??V@S?pA?q?A?QA??'A?(?A?}?>? ?A[YmAWeA???@??C?2SA?3A?WA?X?A@?@?#?@?jFC??8@??A9@h?D@???A???A???A??1C???A?]A???A?&?AЁ?A???A??tA???A1?@@?B?I??lA?ԉA??*Ats?Ak?C?h?AG?nAf??A???A???A?t?A]?LA???Ao/Bc?A?w?AG?A?xkA?^A?AR??A???A?]A3*?A˃?@??QA??{@OҭA???A?=TA?Q?A?? Aɝ?>?c?A[:LAxA?p?@??C?8A??A?u?A?M6A??H@ļ@@??$C?ȟ@1=A;?Z@c?4@??fA??BA??HA???C???A??               	      (               	   (   $   +   %         "            D   ,   M         M   (      /         "               (                     $      '      %         X   ]      A            3   .         *                     !                  '   '      '                     )   3   !         1   )      -                              !               )                     '   /      3                                                                  %                              #               #                                                   '                              +                        Nm??N6N???N?<N???O??cOO?N??O4?ANa2N??KN!??O?O???O?-O??N}b?OH??OA??N?n?N?)?N!?O?
(O?<EO5,dO??NeIDO???P??OD??Of??N?)>N???O??OT??N??N?:?O???O;??OY??O>??Nw??O:?+N?(?N?V?P?+M??>On.O8??N?3hO*h?N??2Op4PO|?,N?aYP9N?<?N?e?N??O??O??N6g?N?]?O?x  ?  ?  ?    ?  V  <    ?  `  8  ?    ?  ?  y  ?  i  u  ?  S    ?  ?  ?  /  <  ]  ?    ?  ?  ?  .  ?  s  |  ?  ?    ?  ?  ?  ?  U  ?    ?  ?  ?  ?  ?  O  
?  	  +  ?  ,  J  	J  ?  !  ?  
??????ě??D????o;D??;??
<u:?o;??
<e`B;?`B<t?<?t?<e`B<#?
<?1<D??<?t?<ě?<?o<e`B<u='??<?/=0 ?<???<?C?=P?`<ě?<???=C?<???<?9X<??h<???<???=o<??h=?w=o='??<???=o=o=C?=\)=C?=?P='??=L??=49X='??=??-=??P=0 ?=]/=<j=H?9=ix?=y?#=?7L=?7L=?\)=?v?(($")5<BFB?5*)((((((????????????????????htty??????????{thhhh?????????????????????????????????????????
%"$
???????????????????????????????


???????????????????????????????????????????????????????????????????????????????????omqsx?????????????to?????
/<DLLH</#	?SSWalz?????????znaUS#/<HOTU[]]H</#????????????????????3/./35BL[glngf]NB:53????????????????????#./<<=>></#|}??????????????||||?????????????????????????!# ?????????????"<CFGB5?BFJNS[grtyxtmgb[NIDBWVZ[hhty?????th^[WW????????????????????{utv}??????????????{????????? 
????)>BHLI=6)??????????????????????????????????????
 #/16<=</#





????????????????????	
#0<DHGE<0#
	????????????????????126ABDOUZQOJB?961111! $/;HTallg`TOH;/)'!yyz{??????????????zy????????	??????? +02)'???429<HQTUH<4444444444???????????????????????????  ??????????	 	
#$,/-'#
				??????&032)??????ghpt????thgggggggggg#/<AHIIGA5)#????????????????????.&')/9<=HOUVUPHD</..?????

???????)5;65) 	)7=BCB86)ghjlt????????????tog?????????????????????????&.)?????????????????????????????????????????????||????????????||||||?????????
??????????? ????xonz?????zxxxxxxxxxx???????

????????????????????????????ŭŹ??????????????ŻŹŵŭťŭŭŭŭŭŭ????????????????????????????????????????????????????????????????????????????????????????????????????????(?,?5???5?4?)?(??????&?(?(?(?(?(?(????)?5?B?P?T?M?=?5?)??????????????????(?2?4?8?4?(??????????????????ûлܻ??????????????ݻܻлûûûû?ƧƳ????????????????????????ƳƧƣƞƠƧ?'?3?9?=?3?/?'?$?????'?'?'?'?'?'?'?'????????	??????	??????????????????¿??????????¿º·½¿¿¿¿¿¿¿¿¿¿?y?????????̿׿ؿѿǿ????????????x?y?v?y?m?y???????????????m?`?G?;?/?,?1?;?G?T?mE?E?E?FFFFF
E?E?E?E?E?E?E?E?E?E?E?Eپ?????????????????????????s?f?[?X?\?h???G?T?T?`?d?h?`?T?G?<?;?@?G?G?G?G?G?G?G?G??)?5?B?N?R?T?N?L?H?B?5?+?)???
??????%?*?1?6?6?4?*???????????????????"?(?*?(?(?????????????????"?*?.?2?.?+?"??	????????	??????N?Z?g?r?g?]?Z?N?J?M?N?N?N?N?N?N?N?N?N?N?????????????????y?m?`?Y?R?S?Z?`?m?y????Ƨ???????????????????ƳƧƚƉƀ?}?ƓƧ?B?O?[?c?h?m?o?h?e?[?O?B?6?-?)?#?%?)?6?B????????????????????????????t?s???????????????????????????????????????????????g?s???????????????????}?Z?N?A?@?E?N?Z?g?????????????"??	?????????????????????ʾ׾??????????????????׾ʾ??????????ž?ìù????????????????????????ýôìçåì?/?<?H?I?N?H?E?<?1?/?.?#?#?'?/?/?/?/?/?/????(?)?)?(??????
???????àìùü??????ÿùìàÓÇÂ?z?}ÇÓÙà??????????????????r?f?_?R?Q?Y?b?f?r?~???????(?.?(???? ???????????????????????????ۺ??????????????H?T?a?h?m?l?i?h?a?T?H?;?(?"???!?/?;?H?)?6?;?B?O?V?Y?S?O?B?6?)????
????)čĚĦĳ????????????????ĿĳĚčąāĀč?=?I?U?X?\?X?U?I?<?0?(?#??????#?0?=?zÇÓÚßÓÇ?z?v?v?z?z?z?z?z?z?z?z?z?z?Ϲܹ????????????ܹϹù????????????ù˹Ͽ	?? ?"?)?"???	???????????????	?	?	?	??????????
????????ּҼѼּݼ??????????	??"?1?:?=?/?"??	?????????????????𻷻??û̻Ȼû???????????????????????????D?D?D?D?EEEEEEED?D?D?D?D?D?D?D?D???????????????????????????????t?t?u?{??????????????
??????????????????????a?n?r?r?r?t?n?j?a?U?H?6?/?(?+?/?<?H?U?a?Z?f?h?n?q?q?f?Z?X?O?P?V?Z?Z?Z?Z?Z?Z?Z?Z???'?.?4?8?2?'?????????????????????4?@?f?r?????x?r?f?Y?U?M?@?4?0???!?'?4D{D?D?D?D?D?D?D?D?D?D?D?D?D?DD{DsDsD{D{?????ֺ????????κ??????~?e?S?M?N?X?r?~?????(?4?A?M?Z?Z?]?Z?M?A?4?(?$??????F?:?5?-?!???????????????!?-?:?F?F?4?@?M?M?Y?]?^?Y?M?F?@?4?4?1?4?4?4?4?4?4???????????
??+?*??
????????????¿?????????????????????????????????~?}???????Ź????????????ŹŷŷŹŹŹŹŹŹŹŹŹŹEuE?E?E?E?E?E?E?E?E?EuEqEoEjEuEuEuEuEuEu?ʼּ??????? ?????????ּʼǼ????ʼʼʼ? R I L = E 0  Y K , @ X = ` ' ; 4 +  & - `  Z 0  : 8 ;  U = a :  f ) . 2 q  2 1 . ( d Y P   > S , D [ S x O ] C # = * .  ?  ?  ?  ;  ?  ?  ?  ?  ?  v    ;  ?    ?  ?  ?  ?  ?  ?  ?  .  ?  5  ?    |  L  X  ?    ?  ?  B  ?  Z  ?  +  ?    ?  ?  ?  ?  ?  ?  ,    ?  ?  ?  ?  ?      ?  /    ?  *  2  ?  ?  !  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  ?  ?       
            ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  Q  "  ?      
          ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  v  Y  =    ?  ?    7  @  H  P  V  S  F  9  (      ?  ?  ?  ?    S  "  ?  ?    ?  ?  ?  ?    !  6  <  6  '    ?  ?    3  ?  b  ?  ?  ?          ?  ?  ?  ?  s  I    ?  ?  ?  g  Z  `  Y  Q  I  l  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  u  ^  ?    ?         ?  ?  ?  ?  ?  ?  ?    E  `  j  l  g  ^  P  @  2    ?  ?  m  8  7  6  0  +  '    ?  ?  ?  ?  a  ;    ?  ?  ?  i  )  ?  ?  ?  ?  {  k  [  J  9  (          ?  ?  ?  ?  h  D    ?  ?  ?  
    
    ?  ?  ?  ?  ?  ?  ?  U    ?  &  ?  ?  ?  ?  ?  ?  ?  ?  ?  G    ?  t  ?  v  X  %  ?  o  ?  M    ?  j  F    ?  ?  ?  T     ?  ?  @  ?  ?  B  '  )  ?     b  L  ^  j  r  v  x  v  q  m  m  k  `  M  .  ?  ?  L  ?  Z  M  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  v  a  M  8  $     ?   ?   ?   ?    C  V  a  g  h  d  W  D  -      ?  ?  }  E    ?  T  ?  2  L  ]  h  o  u  t  m  b  Q  6    ?  ?  ?  D  ?  ?  :  %  ?  ?  ?  ?  ?  ?  ?  ?  l  W  >  #    ?  ?  ?  y  M    ?  S  R  Q  O  N  M  L  G  @  9  3  ,  %         ?   ?   ?   ?          ?  ?  ?  ?  ?  ?  ?  ?  ?  w  c  P  <  (        \  ?  ?  =  t  ?  ?  ?  ?  i  2  ?  ?  	  ?  ?      ?  6  G  V  \  u  ?  ?  ?  z  e  I  )  '  ?  ?  Z  ?  o  ?  ?  
W  
?  %  T  u  ?  ?  ?  ?  ?  h     
?  
d  	?  	9  ?  ?  ?  ?  #  '  *  -  /  ,  '        ?  ?  ?  ?  ?  f  A    ?  ?  <  9  7  5  8  :  =  @  C  D  F  F  @  9  .    	  ?  ?  ?  8  ?  #  m  ?  ?  ?  %  Z  U  :    ?  r    ?  ?  ?  ?  ?  Z  ?  ?  ?  ?  ?  ?  v  N    ?  ?  K  ?  ?  ?  W    ?  [  ?  ?  ?              ?  ?  ?  ?  ?  z  Q    ?  9   ?  ?  8  ?  L  r  ?  ?  ?  x  l  .  ?  ?  &  ?  c  ?       ?  j  ?  ?  ?  ?  ?  ?  ?  ?  g  E     ?  ?  ?  ?  J    ?  :  ?  ?  ?  ?  ?  t  f  \  U  N  G  @  9  -       ?   ?   ?   ?  ?  ?    '  .  +       ?  ?  ?  :  ?  ?    ?  (  ?  .  1  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  Z  $  ?  ?  4  ?  s  s  n  h  c  ]  X  R  M  H  C  ?  :  5  .  "    
  ?  ?  ?    1  X  m  w  |  x  n  ]  D  %  ?  ?  ?  `    ?  \  ?  ?  ?  ?  ?  ?  t  a  K  1    ?  ?  ?  ?  \  !  ?  [  ?  f    o  ?  ?  ?  ?  ?  ?  ?  ?  u  Q  (  ?  ?  v  0  ?  ?  _  U      ?  ?  ?  ?  ?  ?  ?  ?  u  >  ?  ?  k    ?    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  d  F    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  u  d  T  =  $    ?  ?  ?  ?  }  k  ^  J  &  ?  ?  ?  ?  J    ?  ?  9  ?  c   ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  x  l  `  T  G  ;  .  !       ?  T  U  T  R  L  D  8  )    ?  ?  ?  ?  q  I     ?    *  R  ?  ?  ?  ?  ?  ?  ?  e  A    ?  ?  D  ?  w  #  ?  ?  $  ?           ?  ?  ?  ?  ?  i  H  8  '      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  w  D    ?  g    ?  t  ?  p  ,  ?  ?  O  A  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  e  A    ?  ?  X  ?  ?  ?  /  X  }  ?  ?  ?  ?  ?  ?  e  7  ?  ?  M  ?  j  ?  ?  ^  ?  ?  ?  ?  ?  ?  ?  ?  ?  k  E    ?  ?  f    ?  ?  ?  y  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  q  e  Y  M  ?  )    ?  	?  
M  
?  
?  
  "  2  E  H    
?  
r  	?  	w  ?  ?      0  ?  	-  	\  	_  
   
?  
?  
?  
?  
?  
?  
B  	?  	F  ?  ?  ?  ?  ?  (  4  	  ?  ?  ?  ?  g  8  ?  ?  k    ?  8  ?  9  ?  ?    ?  ?    ?    )  "  "       ?  ?  h  -    ?  ?  "  ?  ?  ?  K  ?  ?  x  c  O  C  @  X  ]  U  ?  %  	  ?  ?  ?  p    ?  .  ,  %  "        ?  ?  ?  ?  ?  ?  ?  ?  o  8  ?  ?  ?  n  J  @  7  ,        ?  ?  ?  ?  ?  t  I    ?  ?  N    ?  ?  	1  	F  	;  	&  		  ?  ?  ?  z  9  ?  t  ?  f  ?  `  ?  M  ?  ?  ?  ?  ?  ?    W  $  ?  ?  ?  U    ?  V    ?  ?  6  L  !        ?  ?  ?  ?  ?  ?  p  S  /  
  ?  ?  ~  2  ?  ?  ?  ?  ?  {  f  T  H  4    ?  ?  ^    ?  z     ?  I  ?    
?  
?  
?  
?  
?  
j  
1  	?  	?  	q  	#  ?  E  ?  $  }  ?    c  