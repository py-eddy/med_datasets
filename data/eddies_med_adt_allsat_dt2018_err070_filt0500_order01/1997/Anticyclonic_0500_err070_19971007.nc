CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?`bM????   max       ??"??`A?      ?  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M??   max       P?V?      ?  ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ?T??   max       >#?
      ?  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>Ǯz?H   max       @F>?Q??     ?   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??33333    max       @vrz?G?     ?  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @'         max       @Q@           p  1?   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˊ        max       @??           ?  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ??o   max       >F??      ?  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A???   max       B0Y?      ?  3?   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A?s?   max       B0K%      ?  4?   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ??	?   max       C?x?      ?  5?   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ???,   max       C?w      ?  6?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          j      ?  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          I      ?  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          C      ?  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M??   max       P??;      ?  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ??{???m]   max       ?ڤ??TɆ      ?  :?   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ?T??   max       >#?
      ?  ;?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>?=p??
   max       @F>?Q??     ?  <?   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??33333    max       @vrz?G?     ?  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @)         max       @Q@           p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˊ        max       @??          ?  N?   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D&   max         D&      ?  O?   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ???*0U3   max       ?ڣS&?     ?  Pl   
                     
      N   j         \                        ;         4   H               
            ?   .            -   %      8   e            *   (      _   #      &      N#?qODxFNϏO??NO;?N??vO?&'N?1?N?XPM?YP?V?O'?_N?{P??rNk?N??O?oN%ΗN9nKNͬpN?$P??N??rOG?P]??P2?Oc??N:?=O?LN?N?N???NKPeO?7O??[P??NN??N??N?ѸO???P
?nO"?AO??%PY )N+?N?
'O?0P??O\9PN+G?O?'7O1D?NSҋO^U"N???M???T???D??%   :?o:?o;ě?<o<t?<#?
<e`B<e`B<e`B<?C?<?1<?1<?1<?1<?9X<?j<?j<?j<ě?<???<?`B<?`B<?`B<??h<??h<??h<?<?=+=??=?w=,1=49X=8Q?=D??=T??=T??=]/=]/=aG?=m?h=?o=?o=?o=?C?=?C?=???=???=???=?
==??>J>#?
MJQUahca^UMMMMMMMMMMTW[bgt??????????tg[T44:<@HUacmgaUIH<4444d`^cgt???????????|ld????????????????????45=BFNU[][ZPNMBA:544
	"/7722560"	41167BJHGGDB=6444444?????????????????????????+/33775)???DJnz??)*	????zHDyutvz?????????????zy).6>B=6)??????5[VUM5???????????????????	#)))&03CHIEHU\n????znUH<0?????

	???????RTYaeljca[UTRRRRRRRR????
"!
 ???????//6<HRUXWUH<////////?????"-'???????????? 

???????????
#)/033/+$#
???:<Ohy???????????h[D:???????
#/10)#
????+)(+/3;=HMUY]]ZUH</+????????????????"*6=OX_a\OC6*$)+5BJEB5)$$$$$$$$$$cht????ytnhcccccccc??)-*)?????????????????????????????????????????????????????
#0</#
???????????
,16;/#
?????{nent{?????????????(&#+/<HKUHHH@<7/((((]Yadmnz}~zmga]]]]]]#0IUbdb]TONI<0&???5<BB5+)%????)559;965.)HT\???????????zmaRIHx}?????3)??????~x56BDLOT[\[OLBA<65555aht?????????tkhaaaa?????????????su???????????????|vs????????????????????|?????????||||||||||?????&)*)'?????)+-35<BN[]cfe[TNB5))#""##())#######??????????????????????

????????????????}|?????????????E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E????????????????????????????????????????????????????????????????????????????뾘?????ʾо־ؾվ???????r?b?[?a?z???????? ?!???
?????????????????????????????????ŻŹŭũŭŷŹ???????????	??"?T?a?i?o?p?m?a?T?H?;?"?	?????????????ʼռּ??ּʼ???????????????????????????????????????????þý?????????????????
?<?U?n?x?x?u?n?U?I?<?#???????????????
?Z?????????	?????????????x?u?g?N?7?I?Z??????&?(?*?(???????????ܿ?????????'?)?3?7???3?-?'????????????????????????.?????ƚ?i?E?F?h?zƎƧ????(?/?1?(??????????????????*?6?;?6?6?*?????? ???????????
??#?<?J?L?H?F?<?/?#?
??????????񾥾???????????????????????????????????????????????????ƳƯƳ?????????????????????a?n?z?zÄÅ?z?n?a?Y?U?Q?U?U?a?a?a?a?a?a?n?zÇËÍÇÁ?z?n?m?k?j?n?n?n?n?n?n?n?n???????????????????m?V?K?I?L?h?o?z???????@?L?Y?[?d?e?n?e?Y?R?L?G?@?;?@?@?@?@?@?@??????'?'?(?(?(????????????޽??????Z?f?????????׾??????׾??Z?M?B?C?1?1?A?Z?4?M?T?f?~???????n?Z?M?4????????4?z???????????????????????????z?p?j?l?t?z??"?(?.?'?"???
????????????;?G?T?m?}?????????x?m?`?G???1?/?0?3?7?;?`?g?m?q?n?m?`?Z?]?]?`?`?`?`?`?`?`?`?`?`???????ûĻû????????????????????????????f?h?j?j?g?f?^?Z?M?K?J?J?M?S?Z?c?f?f?f?fE?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?EٿĿѿݿ?????? ??	????ݿѿĿ????????¿?E?E?E?E?E?E?E?E?E?E?E?EiE_EfEiEnEuE?E?E??[?g?t?t?[?N?5???????????)?5?[????????????????????????????????¦²¿??????¿²¦¦Ź??????????????ŹŭťũŭŴŹŹŹŹŹŹ?ּ????????????ּʼ??????????????ʼ?????.?3?,????????¯¦²?????????????N?[?g?t?t?g?[?X?N?F?D?M?NĴ??ĿĶľ????ĿĳĦďā?w?q?p?uĂčĦĴ?????????????պ??????????m?j?o???????ɺ??ѿԿӿѿĿ??????????????????ĿϿѿѿѿѿ????????????????????????????????????????S?`?l?q?y?~??????y?r?l?e?`?Y?S?R?P?N?S???????????????????????Z?P?K?O?Z?s??????????????????"?;?<?>?;?/?"??	???????????6?A?>?:?6?)?$?&?)?.?6?6?6?6?6?6?6?6?6?6???)?0?8?B?E?B?=????????????????????O?[?h?t?wāĈĉĄā?t?h?[?Q?O?M?J?G?O?OĦĦīĳĿ????????????ĿĳĦĦĦĦĦĦĦ???'?)?$??????????????????????????O?\?h?k?q?h?\?O?H?H?O?O?O?O?O?O?O?O?O?O???????????????????????????????????????? C > \ 7 O @ f _ P ? ` > 9 G K O N d I 0 > + j ? \  G 8 , i [ s . d R W j ) 9 . K G W b ? t Z > H Y * 3 N ? * 2  A  ?  
    ?  ?  ?  ?  ?  ?  ?  r  ?    ,  ?  q  l  [  ?  ?  ?  ?  ?    }  ?  `  9  e  L    \  n    ?  ?  ?  ?  o  ?  m  h  ?  ?    ?  ?  ?  d  ?  }  ?  ?  ?  ??o;D??<e`B<???;??
<T??=o<???<?t?=?^5=??F<?<??h=?`B<ě?<?`B=?w<???<?/=#?
=\)=???<?`B=\)=??
=???=]/=C?='??=o=?w=?w=L??=<j=?
==?E?=L??=?O?=m?h=ě?=?Q?=?t?=?G?>!??=?O?=???=??
=?;d=?"?=?1>F??>	7L=>??>n?>#?
B?yB
)BAgB
??B??B?A???B-XBY?B|?B??B?B+?B"B7EBNBx
B??A???B?5BvB?$B#? B+eB?\B#?7B??B-??B0Y?B;B.?B?0B?[BJB??B??B(??B??A???B&,
B?EB??B #?B-?BsB?UB-bEBËB6?B
?%Bn B??B?$BiJB?@B??B??B
dBƙB
??B??B>?A?s?B??B??B?	B>?B7?BDBq?B??B;!BInB??A???B??B>?B??B#??BD?BE?B#B?B-?lB0K%BF?BA?B,7B??BԡB?MBKMB)7?B? A???B%ĞB?$B~?A??B?JB@?BsB-@>Bc?B@?B4?BKB?]B?SB?NB?tB?C?x?A?M=A???AJwRA?=bA??A???@?@A?h>A??A??HA?????	?B??A???A?sXA?fAK??B	A?MA?kwA?b??ϖDA1?AG?A;'A? ?A]?1Ah??Aiy?@???A?fC?r7A|?hC?
?A?:?@Z?%A?? A?؟A?A??JA???A??p@'JSAw?gAq?1A?xA??A?F?A?Q?A?;?A?XA?f?@?ScB??A?_{C?wA??8A?PAJ??AԄ}A?}A?B@???A??BA??AA??
A?|s???,B?|A???A??FA?p?AK)?B?A?a?AȀ?A?}??reA1??AB??A;WA?clA^??Af??Aj??@??:A??C?q?A}O?C??A?a?@\?YA? A???A?bA?%?A???A??@,&?Ay?Aq??A?A?-?A?g}A֔zAӇAAۀ?A??@??B?4A?T?   
                        	   N   j         \                        ;         5   H               
            @   .            -   %      8   e            *   (      `   $      &                  #                  -   I         E         !               )         ;   '                           %   )               +      %   7            )         !               
                              #   C         !                                 5                                 )               +      !   +            '                        
N#?qODxFNzN?O???NO;?N??vO??N?1?N?XO???P??;O'?_N?{O?M?Nk?N??ON%ΗN9nKN?7N@?O???N??rOG?PJd O?ΕOW?N:?=O?LN?N?N???NKPeO?7O%u?P|?NN??N??N?ѸO?<4P ?O"?AO?^?P	??N+?N[n?O?0P
e?OK?FN+G?O??O1D?NSҋO^U"N???M??  
  5    t  `    ?  ?  ?  =  i  {    D  B  ?  ?     	  ?  0  a  [  {  ?  ?  j  ?  '  i  #  ?  ?  ?  
|  !  O    ?  ?  ?  p  
s  ?  ,  8  d  r  `    C  	]  ?  	?    0?T???D??;D??;?`B:?o;ě?<#?
<t?<#?
=?P<???<e`B<?C?=?7L<?1<?1<?/<?9X<?j<???<???=#?
<???<?`B<???=L??<?<??h<??h<?<?=C?=??=?w=?+=8Q?=8Q?=D??=T??=ix?=e`B=]/=ix?=???=?o=?+=?o=?\)=?O?=???=???=???=?
==??>J>#?
MJQUahca^UMMMMMMMMMMTW[bgt??????????tg[T66<HU]`UH<6666666666fflt????????????xoif????????????????????45=BFNU[][ZPNMBA:544
"/5004552/"41167BJHGGDB=6444444??????????????????????????$*...)???KUit~???"????nUKyutvz?????????????zy).6>B=6)???? )5?BB@5)???????????????	#)))&OOQUUWanz~?}|zwnaUOO?????

	???????RTYaeljca[UTRRRRRRRR???


	???????429<HPUUUJH<44444444??????# ?????????? 

???????????
#)/033/+$#
???<>Nh{???????????h[G<??????
 &('#
?????,*(+05<HMSUY\\YUH</,????????????????"*6=OX_a\OC6*$)+5BJEB5)$$$$$$$$$$cht????ytnhcccccccc??),))????????????????????????????????????????????????????
###
???????
+059:4/#
?????{nent{?????????????(&#+/<HKUHHH@<7/((((]Yadmnz}~zmga]]]]]]#0IU[_^YPKKF<0-??5;AB@5)(?????)559;965.)TJJTm???????????zmaT??????????????????56BDLOT[\[OLBA<65555opt????????toooooooo?????????????~xtx???????????????~????????????????????|?????????||||||||||???????')))&???)+-35<BN[]cfe[TNB5))#""##())#######??????????????????????

????????????????}|?????????????E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E???????????????????????????????????????????????????????????????????????????????뾱???ʾϾооɾ?????????p?k?t???????????? ?!???
?????????????????????????????????ŻŹŭũŭŷŹ??????????"?T?a?g?n?l?a?T?H?;?/?"??
???????	??????ʼռּ??ּʼ???????????????????????????????????????????þý?????????????????
?#?<?I?W?d?e?[?U?I?<?#??????????????
?g????????????	???????????}?z?q?]?L?O?g??????&?(?*?(???????????ܿ?????????'?)?3?7???3?-?'???????????ƚƧƳ??????????????????ƳƧƑƆƃƆƌƚ??(?/?1?(??????????????????*?6?;?6?6?*?????? ???????
??#?.?/?<?D?B?>?<?/?#???
????
?
??????????????????????????????????????????????????????ƳƯƳ?????????????????????a?n?w?zÀÂ?z?n?a?]?U?U?U?X?a?a?a?a?a?a?n?zÇÉËÇ?{?z?y?n?m?k?n?n?n?n?n?n?n?n???????????????????????z?l?a?\?_?m?z?????@?L?Y?[?d?e?n?e?Y?R?L?G?@?;?@?@?@?@?@?@??????'?'?(?(?(????????????޽??????Z?f????????׾??????׾??s?Z?M?G?4?4?A?Z?4?A?Z?i?t?w?o?Z?M?A?4?#???????(?4?z???????????????????????????z?r?k?m?u?z??"?(?.?'?"???
????????????;?G?T?m?}?????????x?m?`?G???1?/?0?3?7?;?`?g?m?q?n?m?`?Z?]?]?`?`?`?`?`?`?`?`?`?`???????ûĻû????????????????????????????Z?f?i?i?f?f?]?Z?M?M?L?L?M?T?Z?Z?Z?Z?Z?ZE?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?EٿĿѿݿ?????? ??	????ݿѿĿ????????¿?E?E?E?E?E?E?E?E?E?E?E?E?E~EuErErEuExEE??[?g?t?t?[?N?5?)????????????)?5?[????????????????????????????????¦²¿??????¿²¦¦Ź??????????????ŹŭťũŭŴŹŹŹŹŹŹ?ʼּ???????? ???????ּʼ???????????????*?1?*??????????³®²??????????????N?[?g?t?t?g?[?X?N?F?D?M?NčĦĳľļĵĽ????ĿĳĦĐĂ?x?r?q?wăč?ֺ??????????ݺɺ??????????z?|???????ɺֿѿԿӿѿĿ??????????????????ĿϿѿѿѿѿ????????????????????????????????????????S?`?l?q?y?~??????y?r?l?e?`?Y?S?R?P?N?S???????????????????????????Z?U?O?L?V?g?????	??"?/?;?>?;?:?/?"??	???????????????6?A?>?:?6?)?$?&?)?.?6?6?6?6?6?6?6?6?6?6??????)?.?7?@?B?6???????????????????O?[?h?t?wāĈĉĄā?t?h?[?Q?O?M?J?G?O?OĦĦīĳĿ????????????ĿĳĦĦĦĦĦĦĦ???'?)?$??????????????????????????O?\?h?k?q?h?\?O?H?H?O?O?O?O?O?O?O?O?O?O???????????????????????????????????????? C > N : O @ g _ P 7 ^ > 9  K O . d I 4 < # j ? _  D 8 , i [ i . d @ V j ) 9 . K G X [ ? ^ Z B 4 Y  3 N ? * 2  A  ?  ?  l  ?  ?    ?  ?    ?  r  ?  ?  ,  ?  J  l  [  ?  j  ?  ?  ?  ?  z  ?  `  9  e  L  ?  \  n  u  ?  ?  ?  ?  -  P  m  =    ?  ?  ?  ?  ?  d  ?  }  ?  ?  ?    D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  
    ?  ?  ?  ?  ?  ?  u  M    ?  ?  ?  j  ;    ?  ?    5  '    ?  ?  ?  ?  ?  ?  ?  ?  ?  z  i  [  O  F  E  K  R  ?  ?  ?  ?      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  >  S  a  l  r  s  o  g  Z  F  -    ?  ?  ?  F    ?  W   ?  `  Y  Q  J  C  =  ?  @  B  C  1    ?  ?  ?  {  X  5     ?      ?  ?  ?  ?  ?  ?  ?  ?  ?  o  d  Z  @    ?  ?  ?  P  g    }  n  ]  N  ?  -    ?  ?  ?  n  Z  X  4  ?  ?  C  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  p  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  l  W  A  *    ?  ?  ?  ?  ?  d  ;  ?  ?    !  0  :  ;  !  ?  ?  ?  Q    ?  w    ?  ?  ?  @  $  V  h  a  T  2  ?  ?  f       ?  ?  ?  L  ?    (  ?    {  t  j  ^  O  >  *    ?  ?  ?  ?  R    ?  ?  D    ?  ?      ?  ?  ?  ?  ?  ?  q  O  ,    ?  ?  ?  W  0    ?  ?  Z  ?  ?  ?  ?    !  *  8  B  B  4     ?  ?  \  ?  ?  ?    B  4  &    	  ?  ?  ?  ?  ?  ?  ?  u  _  F  .     ?   ?   ?  ?  ?  s  e  \  R  G  <  1  #      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  f  l  ?  ?    ,  T  ?  ?           	            ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  	  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  p  _  M  <  %     ?   ?   ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  w  R  ?  .  ?      "  .  .  *  $      ?  ?  ?  p  .  ?  ?  (  ?  X   ?  ?    2  J  X  a  \  K  .    ?  ?  m    ?  I  ?  $  R  S  [  V  Q  K  F  A  ;  7  3  /  *  &  "      ?  ?  ?  ?  ?  {  g  S  @  3  %    
  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  `  ?  ?  ?  ?  ?  ?  g  -  ?  ?  p  D    ?  ?  <  ?  _  ?  m  ?  G  }  ?  ?  ?  ?  ?  ?  ?  J  ?  ?  4  ?  i  ?  @  w  ?  O  i  e  _  U  I  9  '    ?  ?  ?  m  +  ?  ?  A  ?  r  2  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  s  f  b  a  `  _  '      ?  ?  ?  ?  ?  ?  ?  v  e  Y  O  K  D  6        i  c  ]  W  Q  J  D  >  8  2  *            ?   ?   ?   ?   ?  #          ?  ?  ?  ?  ?  ?  t  S  2    ?  ?  ?  }  W  ?  ?  ?  ?  ?  ?  ?  ?  ?  s  Q  &  ?  ?  ?  =   ?   ?   ?   ?  ?  ?  ?  ?  ?  v  b  N  9  $    ?  ?  ?  ?  ^    ?  p  $  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  v  l  c  Y  I  0    ?  ?  e  ?  	.  	u  	?  
,  
S  
l  
z  
v  
^  
1  	?  	?  	)  ?  ?  ?  ?  ?  i  !    
     ?  ?  ?  ?  ?  ?  ?  ?    N    ?  p  ?  R  ?  O  I  D  ?  :  6  1  -  !    ?  ?  ?  ?  ?  g    ?  k      ?  ?  ?  ?  ?  l  O  ;  7    ?  ?  o    ?    e  ?   ?  ?  ?  ?  ?  ?  s  ]  H  4      ?  ?  ?  ?  ?  ?  r  P  .  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  w  W  6    ?  {    P  M  5  ?  ?  ?  ?  ?  ?  S  !  ?  ?  ?  I    ?  ?  \  O    ?  1  p  f  W  D  /    ?  ?  ?  ?  ~  Z  3    ?  ?  ?  ]  3  R  
l  
n  
Z  
:  
  	?  	?  	5  ?  	  ?  ?  O  ?  v  ?  	  ?  :  %  
?    ^  ?  ?  ?  ?  ~  M    
?  
H  	?  	K  ?  ?  )  '  ?  3  ,      ?  ?  ?  ?  ?  }  C  
  ?  ?  e  1  ?  ?  ?  V        /  7  1  &    ?  ?  ?  ?  ?  ?  z  b  I  /    ?  ?  d  A    ?  ?  ?  k  3  ?  ?  n  <    ?  ?  ?  ?  ?  x  N  g  q  e  O  5    ?  ?  ?  u  ;    ?  ?    G  ?  ?    ?  ?  \  Q  >  )    ?  ?  ?  r  E    ?  ?  !  ?  ?  ?    ?    ?  ?  ?  ?  ?  k  R  :  #    ?  ?  ?  ?  ?  ?      7  	  B  0    ?  ?  g    ?  T  ?  V  ?  ?    
#  ?  ?  ?  ?  	]  	P  	<  	%  	
  ?  ?  ?  Z  !  ?  ?  F  ?  w  ?  k  n     ?  ?  ?  h  F  $    ?  ?  ?    T  (  ?  ?  ?  _  $  ?  ?  s  	?  	?  	?  	?  	?  	?  	e  	9  	  ?  ?  P  ?  ]  ?  ?  ;  D  (  ?    ?  ?  ?  ]  /  ?  ?  ?  j  7    ?  ?  :  ?  ?  `    ?  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0