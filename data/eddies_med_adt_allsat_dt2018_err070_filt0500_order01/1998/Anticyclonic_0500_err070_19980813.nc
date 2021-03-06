CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?`bM????   max       ?ɺ^5?|?      ?  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M??   max       P??2      ?  ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ????   max       =???      ?  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>?Q???   max       @F,?????     p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??z?G?    max       @v?33334     p  (?   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @0         max       @Q?           l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @??        max       @???          ?  1?   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ?u   max       >??      ?  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A??   max       B0lN      ?  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A?w   max       B0@?      ?  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =?9?   max       C???      ?  4?   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =r?q   max       C???      ?  5?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ?      ?  6?   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          ;      ?  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          3      ?  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M?cn   max       P??      ?  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ???+J   max       ??$?/?      ?  9?   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ??t?   max       =??m      ?  :?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>?Q???   max       @F,?????     p  ;?   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??z?G?    max       @v??\(??     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @)         max       @N?           l  L?   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @??        max       @?'           ?  L?   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E\   max         E\      ?  M?   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ?nOv_ح?   max       ??S&??     ?  N?         	   5      C         *   H                     
   &   	   O   R   
            !   
                  
   
         #   ?   T   C         >      .   2   
               6   6   N?TNN??N	ƬP?{tN5W?P?YO???O;E?OChPH??NƇ?O4{N?'?O),oNa?wNe\NMe?PC?Nrk-P??2PK7?O?f?O%M?OiŧOF??O??N?9?O_y?N???N??fN?|pN?KN?^NaD?O?N??O???O??O??P7l?O?KN?0O???O???O??mO??^N?O?PN??NN%M3Nz??Ox?PO???M???????ě??ě??o??o;o;o;o;?o<o<#?
<#?
<T??<?C?<???<???<?1<?9X<ě?<???<???<?/<?`B<??h<??h=o=o=o=o=C?=\)=?P=??=8Q?=<j=D??=H?9=L??=P?`=T??=T??=m?h=}??=?%=?+=?7L=?O?=??=??=ě?=ě?=???=???=????????????????????????????????????????????????????????????????????#/ai_<65/#
???? 	
"///"!	        ???):KOUSB5)????????????????????????????????? 

??????pmklqt???????????xtp?????
/<Kbd_H)#???4/08<HKU]_UTH<444444),69960)$????????????????????????????????????????rv{????????{rrrrrrrr(&*6CDOSOGC60*((((((?}}~????????????????????phdm????????????QUWanz||znnnaUQQQQQQ)J|???ztB506BO\hz|e[OB/hjebcmvz?????????zmh????????????????????|}?????????????????|??????
!#'#
?????#)5<FNg??}|sgNB5)?????????????????????????????
???????,'#&0<GF?<40,,,,,,,,?????????????????????)*//+)
????????????????????????????????????????#08840#?????
 "!
????????????????????????????????????????????????
  
??????????6BOYg[PHB6)
???tg[N<13O[g??????????
)=BNUOC?6??ajmqxtma`aaaaaaaaaaa70#
#0?FGFB@7????????*+0)????????????????????????????????????????????????

???????????UV^lz???????????zmaU????????????????????#


#'%#########???????????????????????

???????????????????????&)0666)'?B?J?N?Y?W?N?B?5?)?$?)?.?5?=?B?B?B?B?B?B??????????????????޹??????????????????ùŹ͹ù????????????????????????????"?;?T?????????????z?L?;?"???????"?T?_?a?e?c?a?a?`?Y?T?J?K?T?T?T?T?T?T?T?T?<?U?nŕśŘŇ?b?U?#?
???????????????#?<??(?5?N?Z?g?????????????N?A?9?%???????????????????????????????x?n?s?x?~?????a?n?zÇÓÙáÓÑÇ?z?n?a?U?O?N?O?U?Y?a??T?`?y?????????}?`?.?"???????????	????????????????????????????????????????????	?????????????????????	?????ʾ׾پܾ׾ʾ??????????????????????????"?/?;?H?T?a?m?x?m?d?a?L?H?;?7?/?%??!?"?????????????????????????????????y???????????}?y?s?m?k?k?m?t?y?y?y?y?y?y???????????????????????????????????? ????(?4?M?f?v?????|?s?f?Z?4????zÆÇËÎÑÇ?z?u?n?n?n?t?w?z?z?z?z?z?z?"?;?H?S?V?T?H?/?"?	?????????????????	?"???׿	?&?'?&?!??	???׾ʾ??????????????????????????"?&?"??	???????????????????A?M?Z?f?f?k?m?f?e?Z?M?A?4?0?)?-?4?<?A?A??????(?4?=?A?H?L?A?'???????????????A?M?U?Z?^?f?h?d?Z?V?M?>?4?0?-?0?4?<?;?A?`?m???????????????????m?`?T?I?B?A?D?M?`???????????ĿĿ????????????????????????????????????????????????x?o?g?f?c?e?r??f?r???????????r?n?f?a?f?f?f?f?f?f?f?f?
???"?#?#?#??
???????????
?
?
?
?
?
???????????????????????|????????????????/?<?H?Q?H?E?<?5?/?.?,?.?/?/?/?/?/?/?/?/ÓÜàìñùüùóìàÓÐÓÓÕÓËÓÓ???????? ???????޼ؼ?????????????????????(?5?A?A?N?Q?Q?N?A?5?(????	????²¿????????¿¶²¦¤¤¦¯²²²²²²FF1F=FVFdFoFyF}FuFcFJF=F/F$FFFE?FFD?D?D?D?D?D?D?D?D?D?D?D?DxDvD{D?D?D?D?DƼ????????żϼμ˼?????????r?m?i?j?o????????????????????)?O?U?W?H?@?6???????3?@?Y?~?????ֺ⺽???????~?m?Z?S?@?/?+?3??????????ùíøù?????????????????????ź??????????????ֺ??????????????ֺɺ??`?l?y?????????????????y?l?`?S?P?M?O?S?`?O?[?tāčĚĦĳĺĽĽĳĦč?h?[?O?A?L?O?o?{ǈǖǞǡǤǡǔǈ?{?o?i?b?^?]?_?b?i?o?h?tāąĄā?t?m?h?f?h?h?h?h?h?h?h?h?h?hŠŭŹ????????????????ŹŵũŤŝŘŕŔŠ?
?	???????????
??????
?
?
?
?
?
?
?b?]?b?n?n?n?{?}Ł?{?n?b?b?b?b?b?b?b?b?b????????????????????????????????????????E?E?E?E?E?E?E?E?E?E?E?E|EuEiEhEjEuE?E?E????ûлܻ????ܻѻ????x?l?h?m?x???????????S?_?g?_?\?S?G?F?F?D?F?Q?S?S?S?S?S?S?S?S A + V Q t $ f 2 ; 6 * H P : N ] Z " b 9 / u   q e : C 5 5 5 ) S Q ( I _ ` / * @ ? N ; r F & I i \ 9 b , j z  ?  ?  5  m  ?  `  ?  ?  ?  ?  ?  s  ?  {  ?  ?    ?  ?  ?  ^  ?  Z    ?  H  ?  ?  ?  ?    J  ?  r  1  ?  L  ?  ?    !  3  l  ?  ?  ?  :  ?  ?  D  ?  ?  ?  6?u;?o;o=P?`;D??=?O?<?9X<ě?=8Q?=???<?<?o<?C?<?/<??h<?1<???=q??=+=???=?/=?P=q??=,1=8Q?=??='??=u=?w=?w=@?=,1=@?=]/=m?h=??=???>??>
=q=???=???=?%=??m=??=?S?=??h=???=ȴ9=?"?=??=??`>??>!??=?/Bk]B?mB`eB?/A??B??B?B#?B
i?B?B?B?CB1?B?@B)?B0lNB?.ByBu?B?\B?2A??{B?B?jB#?6B$?BV-B"?6B%??BC?B?B 1eB!??B%f?B?zB*?BI?B?B:B	?CBE?A??VB%H6B-?EBB?BW?B?lB 1IB?KB_aBe?B? B?yB;?B?B??BCYB?!A?wB%?B;?B"??B
@?BsuB5.B<HB>?B??B)=?B0@?B<zB?B?2B,#B?1A???B??BJB$>OB6zB?MB"??B%??B[?B??B ?B"?B%CMB?PB??B\?B??B??B
?TBK3A?~B%@jB-?OB??BD?BA?BB?WBB?B??B??B@?B9?A?e??6?=?9?A?dA??^A?gmA???@?b?A???Af??A?ܚA??AO??A?z?@Z??Am?+@?@hA;?'AȧoA?@RARa?A??*A=?A3?TA<^Ak??At@?7V@?=A???Ar=A?^?A?ĞA?AA?ʆA?+?C???C??W@?ԐA?,K@??A?*?@@&EAzsA??ZBfXAܤ?A???A?)?A???A?h?C??@??>@???A?}?B?=r?qA?ƾA??:A늇A??@???Aǋ&Ai?A?z?A???AM A??Y@\?Al??@?8A;??AȐA??kAR??A?tA=PA/NA=?Aj??At?@??{@? ?A?TVAr?A?lrA??A?A???A??C???C??@? ?A՛>@?A΀?@D&?A?_A޺?B=aA܀?A??A?~~A?~HA???C??@???@??>         	   6      D         *   H                        &   
   O   S                "   
                  
   
         #   ?   U   C         >       .   2                  6   7               ;      7   %         -                        (      9   1               %                                 !      !   ,   -   
                                 #                     /   !         !                        !      3   #                                                         ,      
                                    NM??N??N	ƬOR?N5W?PnZ?O?ްNxa?N??!O??IN??%O4{N?'?N???Na?wNe\NMe?O???NA?PP??O?<oO?f?N??3OQOF??Ol{N?9?O9­N???N??fN??GM?cnNZA?NaD?O?NF$?OK??O/	EO??'P7l?OXw N?0O???Oy??O?СO??^N?O?PN??NN%M3Nz??OM8?Oe??M??  ?  a  |  ?  T  ?  Y  p  ?  ?  ?      B     ?    ?  Y  ?  ?        ?  ?  M  ?  ?  ?  ?  ?  ?    ?    i      
Z    ?  	z  ?  	1  ?  ?  b  ?  ?  W  
>  	?  弓t??ě??ě?<??h??o<#?
;?o<T??<u=t?<e`B<#?
<T??<??
<???<???<?1<?<???=\)=]/<?/=t?=+<??h=49X=o=\)=o=C?=t?=??=#?
=8Q?=<j=L??=aG?=??m=?hs=T??=q??=m?h=}??=?7L=?\)=?7L=?O?=??=??=ě?=ě?=?
==?S?=???????????????????????????????????????????????????????????????????
#*-)%"
???? 	
"///"!	        ?????)5BJNQNB?????????????????????????????? ????????????spost?????????????ts??
#/AHOQPH</#
?723<HUYYULH<77777777),69960)$????????????????????????????????????????rv{????????{rrrrrrrr(&*6CDOSOGC60*((((((?}}~????????????????pqu???????????????vpRU]anz{{znaURRRRRRRR
>o~???~tgN5)
)6BO^bd]YOBhjebcmvz?????????zmh????????????????????????????????????????????
!#'#
?????''*.5N[gntupfVNB5/)'??????????????????????????????

???????,'#&0<GF?<40,,,,,,,,???????????????????? ))..*)????????????????????????????????????????#08840#?????
 "!
??????????????????????????????????????????????????

?????
)6BNPLF;6)
???tg[N<13O[g???????
%)26BHIL?;6)ajmqxtma`aaaaaaaaaaa70#
#0?FGFB@7??????#*%???????????????????????????????????????????????????

???????????UV^lz???????????zmaU????????????????????#


#'%#########????????????????????????

	??????????????????????&)0666)'?5?B?N?V?Q?N?B?5?+?0?5?5?5?5?5?5?5?5?5?5??????????????????޹??????????????????ùŹ͹ù????????????????????????????T?a?m?q?v?v?s?m?a?T?H?;?/?,?+?/?2?;?H?T?T?_?a?e?c?a?a?`?Y?T?J?K?T?T?T?T?T?T?T?T?#?<?I?U?nŋŒŏŅ?l?U?<?#?
??????????#?(?5?N?X?g???????????g?N?A?=?5?(????(?????????????????????????????????????????a?n?zÇÉÓÓÉÇ?z?q?n?a?^?U?T?T?U?a?a?m?y?????????}?`?T?;?.?"????"?:?J?T?m?????????????????????????????????????????????	?????????????????????	?????ʾ׾پܾ׾ʾ??????????????????????????;?H?T?Y?a?f?a?^?T?H?;?0?/?)?/?7?;?;?;?;?????????????????????????????????y???????????}?y?s?m?k?k?m?t?y?y?y?y?y?y???????????????????????????????????4?A?M?f?z?~?y?r?f?Z?4?(???????(?4?zÂÇÈÍÊÇ?z?w?o?u?x?z?z?z?z?z?z?z?z?	?"?;?L?Q?H?/?"??????????????????????	?ʾ׾?????????	?????׾ʾ?????????????????????????"?&?"??	???????????????????M?Z?Z?e?f?i?f?_?Z?M?A?;?4?1?4?7?A?L?M?M??????(?5?A?B?A?4?(????????????????A?M?U?Z?^?f?h?d?Z?V?M?>?4?0?-?0?4?<?;?A?`?m?y?????????????}?y?m?`?T?O?M?P?T?[?`???????????ĿĿ????????????????????????????????????????????????|?s?l?f?i?r?w??f?r???????????r?n?f?a?f?f?f?f?f?f?f?f?
???"?#?#?#??
???????????
?
?
?
?
?
???????????????????????}?????????????????/?<?H?K?H?C?<?3?/?.?-?/?/?/?/?/?/?/?/?/àìïùûùòìàÓÒÓÖÙàààààà???????? ???????޼ؼ?????????????????????(?5?A?A?N?Q?Q?N?A?5?(????	????²¿????????¿²¦¥¦°²²²²²²²²FF$F1F=FVF]FoFuFyFoFcFVFJF=F1F'FFFFD?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D??????????ǼǼż?????????r?p?q?v????????????????????????)?O?U?W?H?@?6???????e?r?????????????º̺????????~?s?k?b?a?e??????????ùíøù?????????????????????ź??????????????ֺ??????????????ֺɺ??y???????????|?y?l?`?X?S?P?N?Q?Q?S?`?l?y?tāčĚĦĳĹļĻĳĦč?~?t?h?P?G?O?[?t?o?{ǈǖǞǡǤǡǔǈ?{?o?i?b?^?]?_?b?i?o?h?tāąĄā?t?m?h?f?h?h?h?h?h?h?h?h?h?hŠŭŹ????????????????ŹŵũŤŝŘŕŔŠ?
?	???????????
??????
?
?
?
?
?
?
?b?]?b?n?n?n?{?}Ł?{?n?b?b?b?b?b?b?b?b?b????????????????????????????????????????E?E?E?E?E?E?E?E?E?E?E?E?E?E?E~EuEmEuEvE??ûлܻ߻߻ڻл˻û????????????????????ûS?_?g?_?\?S?G?F?F?D?F?Q?S?S?S?S?S?S?S?S 5 + V  t   h ( 0 V & H P " N ] Z  i 4 $ u  m e 6 C 4 5 5 ' Y = ( I N T   @ k N ; p F & I i \ 9 b ( L z  \  ?  5  ?  ?  ?  t  ?    ?  ?  s  ?  ?  ?  ?    ?  ?  !  ?  ?  ?  ?  ?     ?  ?  ?  ?  ?  7    r  1  V  ?  q  0    
  3  l  ?  ?  ?  :  ?  ?  D  ?  ?  ?  6  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  E\  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?     a  ]  Y  R  I  ;  ,      ?  ?  ?  ?  i  L  /  %      ?  |  y  v  p  i  c  a  ^  G  -    ?  ?  ?  r  ?    ?  ?  f  r  ?  
  ;  ]  `  Q  A  K  o  ?  ?  ?  ?  w  8  ?  {  ?  )  T  [  b  j  q  v  u  t  s  r  o  j  f  a  \  X  S  O  K  F  ?  ?  ?  ?  ?  ?  z  m  b  R  :    ?  ?  k  (  ?  ?  &  ?  D  T  Y  R  B  /      ?  ?  ?  ?  ?  ?  ?  w  J  I  ?  ?  9  >  @  A  A  A  C  ?  A  V  m  f  Z  E  .    ?  ?  ?  ?  &  S  y  ?  ?  ?  ?  ?  v  M    ?  d  ?  |  ?      ?  ?  ?  	  ?  g  ?  ?  ?  ?  ?  ?  ?  j    ?  ?  M  ?  ?  .    ?  ?  ?  ?  ?  ?  ?  ?  m  U  8    ?  ?  ?  ?  b  i  $  ?           ?  ?  ?  ?  ?  ?  |  c  J  /     ?   ?   ?   ?   ?        
      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  3  4  5  3  3  <  >  2  $      ?  ?  ?  ?  ?  ?  ^  (   ?    ?  ?  ?  ?  ?  |  b  I  .    ?  ?  K  $    ?  ?  j  ,   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?      p   a   R   C   4   %    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?    1  K  d  }  ?  ?  ^  t  ?  ?  ?  ?  ?  ?  ?  n  Q  +  ?  ?  ?  a    ?    ?  E  N  X  W  U  M  B  <  8  7  :  8  /  &      ?  ?  _  ?  ?  ?  ?  ?  ?  |  3  ?  }    ?  i  !  ?  ?  ?  d  ?    <  ?  ?    N  q  ?  ?  ?  ?  j  B    ?  ?  2  ?  ?  +  ?   ?    ?  ?  ?  ?  ?  w  r  l  f  `  [  H  +     ?  ?  c     ?  ?  ?  ?          ?  ?  ?  ?  Y    ?  r    ?  i    ?    ?  ?  ?        ?  ?  ?  ?  ?  v  =     ?  ?  ?  ?  s  ?  ?  ?  ?  l  a  i  }  ?  ~  b  E  '    ?  ?  ?  3  ?  4  {  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  N    ?  w  H    ?  .  ;  M  9  $    ?  ?  ?  ?  ?  ?  l  R  6      ?  ?  ?    \  ?  ?  ?  ?  ?  ?  ?  ?  ?  h  P  6    ?  ?  }  E  !    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  y  o  c  X  J  :  )    ?  ?  ?  ?  ?  ?  ?    x  q  j  c  ]  W  W  [  _  d  E    ?  ?  ?  ?  ?  ?  ?  ?  ?  s  d  O  5    ?  ?  ?  u  C    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  r  c  T  F  7  +  !      ?  ?  ?  ?  ?  ?  ?  ?  ?  v  ^  =    ?  ?  ?  ?  ?  ?               ?  ?  ?  ?  ?  ?  j  D    ?  ?  ?  _  )   ?  ?  ?  ?  {  f  Q  ;  #    ?  ?  ?  ?  ?  x  S  '  ?  ?  $  ?  ?  ?    ?  ?  ?  ?  q  Q  +    ?  ?  ~  T  -  ?  ?  _  -  O  b  h  h  f  a  R  3    ?  ?  ?  ?  ?    ?      ?  ?  ?  ?  i    ?  ?      ?  ?    S  j    +  ?  ?  ?  	H  
  
?  
?  
?  
    
?  
?  
?  
x  
2  	?  	`  ?  1  r  ?  i  ?  *  
Z  	?  	?  	J  ?  ?  `  
  ?  Y    ?  U  ?  ?  ?    l  ?  ?  ?  ?  ?  ?  ?      ?  ?  ?  ?  x  F    ?  c  	  ?  ?  1  ?  ?  ?  ?  ?  ?  ?  ?  ?  {  o  d  Y  O  D  :  /  %      	z  	q  	c  	L  	(  ?  ?  ?  z  B    ?  ?  @  ?  Z  ?  ?  Q  ?  ?  ?  ?  ?  ?  ?  i  <    ?  ?  -  ?  ?  '  ?  y  (  ?  ?  ?  	)  	0  	$  	  ?  ?  	  ?  ?  ?  ,  ?  I  ?  7  ?  ?  %  ?  ?  ?  `  J  *    
?  
?  
2  	?  	f  ?  ?    p  ?  ?  ?  ?    ?  ?  ?  i  K  1    ?  ?  ?  ?  ;  ?  ?  ?  O    ?  ?  f  b  F  %    ?  ?  x  P  #  ?  ?  ?  W  (  ?  ?  8  ?  ?  '  ?  ?  _  ;    ?  ?  ?  a  7    ?  ?    K    ?  w  )  ?  ?  ?  ?  ?  o  \  G  -  (  B  >    ?  ?  ?  ?  \  0     ?  W  K  ?  3  #      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  t  b  
/  
/  
=  
1  
  	?  	?  	?  	K  	
  ?  u    ?  L  ?    [    ?  	p  	?  	?  	?  	?  	?  	?  	}  	N  	  ?  ?  F  ?  u  ?  J  e  ?  H  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?    s  f  Y  M