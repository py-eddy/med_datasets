CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?h?t?j~?   max       ??fffffg      ?  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N??   max       PA݊      ?  ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ??j   max       =?x?      ?  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>???Q?   max       @F?Q??     ?   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??z?G?    max       @ve???R     ?  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @0         max       @P?           p  1?   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @?e        max       @?H?          ?  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ??o   max       >???      ?  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A?#   max       B/??      ?  3?   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A?d?   max       B/?o      ?  4?   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       <?C?   max       C?	      ?  5?   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ??u?   max       C??5      ?  6?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ?      ?  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          1      ?  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      ?  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N??   max       P?C      ?  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ??!?.H??   max       ?ߜ?ߤ@      ?  :?   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ??9X   max       >+      ?  ;?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>?33333   max       @F?Q??     ?  <?   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??\(?    max       @ve???R     ?  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @#         max       @M?           p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @?e        max       @?@          ?  N?   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A[   max         A[      ?  O?   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ???N;?5?   max       ?ߚkP??|     ?  Pl                  (   !   
      	   ?            !   1      ?      )   	                        F         g   7      7      B                           	      	   &      !      	   !   ?   N??lN???OZ??OR96O???O??HPO?N??*NH??Oa??PA݊O?N6?|O???P??O???O?k?P"'?N?lO?7}N???N'N_N??O9'?NΒ?N?K?N??O̪iP+?VO4>,OyɞPa]O΁^OQ?XP??O?WP
*@O)N?r?N?`?O'?}O2?5N7?O?|O??NUU?O ՃN???O5??O<?Ol?O??NV??O??P3??O???j??9X?t??o;o;??
<#?
<#?
<?o<??
<?1<ě?<???<?`B<?`B<?`B<?`B<?`B<???<???<???=o=o=o=+=??=?w=#?
='??=,1=0 ?=49X=<j=H?9=H?9=P?`=T??=Y?=Y?=]/=]/=q??=u=y?#=?%=?o=?o=?o=?C?=??-=??-=???=??
=???=? ?=?x?rort???????trrrrrrrr????????????????????#/<FHNUW^^UH</#????????????????????)5@NSX[^B5)
vonrr{?????????????v?/;HT^gaTF;/"
??hmz??????????zmhhhhBBINV[^^[NBBBBBBBBBBvuvwz?????????????zv???????????????????????????????????????????????????????????wwx??????????????|w????????????????????????????????????	)BN[g}}tgZ5???????????????????????????????????????"(*0<HUap|?znaUC/'"????????????????????
)/*)







kekmnwz}zxsnkkkkkkkk% )6BOY]^]][WOGB6)%????????????????????????????

??????!*36CDHDC60*'" #/<E_afYNF@</$"????#/<NFKUH;9)
????????????????	)6BEF><6$?????????????????????????????????????????????????????????????????5<AA>5)	???	
#0<=<;710(#"
	?????)/:<<5)????????
#$$#
???		
#//33/,#
		!##/<@FGE></*#!!!!UPRUX[^ht????tjhb[U???)/16860)$????

???????????????
#%&
???????#)25BDMPQPNHB5#??????????????????????????????????????????$)-,)????????????????????????????????????????????SOV[ahttu?????{th[SS????????????????????????????????????????????? 
"###
????)5N[u?????t[B)		
 #//97/#
	?zÇÓÙÛÔÓÇ?z?x?u?x?z?z?z?z?z?z?z?z?{ŇŔŠŭŲŭŬŠŔŏŇ??{?v?y?{?{?{?{???????????????????????????????????????????????????????????????????????T?`?k?t?v?u?m?a?T?G???;?5?2?2?5?;?@?G?T?O?[?h?tāčĚğĥĢĚĔā?t?[?S?F?<?A?O?;?T?a?h?q?s?o?h?a?T?G?$?	?????????	?"?;?a?b?e?m?w?z?????z?t?m?a?^?Z?Z?[?a?a?a?a?n?{ŇōŇŃ?{?n?i?d?n?n?n?n?n?n?n?n?n?n?n?{ŇŔśŭŲŶųŠŜŏŇ??s?o?q?s?j?n???????*?4?/?+? ????ֺ????????????ɺ??F?:?3?:?-?!??????????????!?-?:?F?F?/?<?C?>?<?/?#??#?%?/?/?/?/?/?/?/?/?/?/?\?h?uƁƈƌƉƅ?{?u?h?\?C?8?2?6?;?E?O?\?(?5?A?I?]?k?p?n?Z?5???????????
??(????????/?1?*?$????????źŷż???????߿??Ŀݿ??????տ˿ȿ??????????????????????????(?-?4?@?M?A?(?齷???????????Ľݽ????????????u?s?n?m?s?v??????????5?A?N?Z?_?_?`?Z?N?A?5?(??????
???N?R?Z?`?a?`?Z?W?N?A?5?0?1?5?7???A?I?N?N?f?s?s?s?r?l?f?Z?Y?X?Z?c?f?f?f?f?f?f?f?f?a?n?zÇÈÇÅ?z?p?n?a?`?a?a?a?a?a?a?a?a?ʾ׾????????????׾ʾ??????????????žǾʿm?o?y???????????y?m?`?^?W?W?`?h?m?m?m?m???ûлٻܻ??ܻٻлû????????????????????m?y?????????????????y?m?k?b?`?_?`?f?m?m??(?A?M?f?m?}??s?Z?U?A?(??????????N?[¦³²¼½²¦?t?B?)???5?N?????????????|?y?s?l?j?`?[?Y?Y?_?`?l?y???:?S?x???x?p?_?[?U?N?J?F?:?-????!?-?:?????ù????????ܹù??????x?k?m?z??????ù????????????????????ùâÈÊ×àìù?H?U?a?n?k?c?U?L?<?/?#?????#?(?/?<?H¿?????????????²?t?c?_?c?t?{£¿???????????????????ּʼü??????ɼּ????#?0?<?I?O?U?S?L???0?
??????ĸ?????????#?????	???"?#?!?"?'??	?????????ھ??????T?`?m?n?q?m?j?`?_?T?G?>?;?0?;?C?G?S?T?T?=?I?V?b?b?i?h?b?V?I?=?0?&?,?0?6?=?=?=?=?Y?f?r??????????????????????u?f?^?Y?M?Y?Y?e?r?~????????r?e?[?Y?T?L?B?@???L?P?YD?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D????	???"?%?"?#?"???	??????????????????????????$?1?5?$?????ƳƧƚƕƕƧ???
???????
???????????
?
?
?
?
?
?	???"?%?/?0?/?%?"???	???????????	??????????????????????????????????????伱?ʼּڼ߼ۼּӼʼļ???????????????????????????#?(?4?4?5?(????????????????3?@?L?Q?Y?^?e?e?e?e?Y?L?@?8?3?/?-?/?3?3?M?V?Y?`?Y?Q?M?@?4?'???
?
???'?4?@?M?????????????????x?l?f?l?x?y????????????E?E?E?E?E?E?E?E?E?E?E?E?E?E?EvEvE{E}E?E??B?[?tāĐĚğēĐā?t?[?B?6?0?"?? ?.?B??*?6?C?F?O?W?O?M?C?B?*???????? ? = * K C ) ) ? $ 7 * 1 V $ . ( s > w ) R Z U K 1 @ - A = X ` 2 0 S b # M 7 B ( _   a | ] S 9 : E S F b X < ) (  ?  ?  ?  ?  +  S  ?    X  ?  6  f  -  G  Z    ?  (  ?  ?  4  [  G  ?  ?    ?  ?  $  ?  A  ?  ?  ?  ?  H  ?  T  ?    ?  ?  U    q  ?  *  ?  ?  ?  0  y  q  M    E?D????o<?1;ě?<?9X=49X=,1<???<?9X<??h>?-=L??=C?=H?9=u=??-=8Q?=?Q?=\)=?t?=?w=\)=C?=0 ?=#?
=u=L??=?7L=?G?=Y?=aG?>?+=???=???=???=??==?o=?7L=?C?=?t?=???=?7L=?E?=?-=?t?=?hs=??=??=Ƨ?=?;d=ȴ9=?9X=??>???>\)B
N?BCB??BasB?FB ?"A?#B uBQuB ??B??B _?B?6B#LB??BT?B9:B"&?B)?B??B_LB3yB?IB*B?bB#g?B/??B?B??B-;mB??B??B.BR=BQ?B%C&B??B??B??BdB֞B?mB?MB?B??B?7B?=BtPB}?B?ZB??B"?B,?#B??B?zB??B
G?B? B??B9%B??B ?jA?d?B ??B??BBĦB D?B??BF?B`FBRVB7?B">6B?&B??B??BI?B??B4?B??B#??B/?oB??B?B-+0B&SB?EB<?BA?BB?B%@BơB?BB?BA?B0?BcHB? B4?B??B??B??B??BA&B?B>%B:-B,??B?[B?4B7?A?t?A?]XA??A???Af܂A?Q?A??xA?vkA?6>A?߿@E?T@dљA¦IB?oA??A??!Aw}?A.?6AD?DA?n?A?)0A@?YA??ASJ?Ak_C@?@?Am?A:?A???AE?@??C<?C?Aθ@A?3A??9AP,A??VAY??AgBZ?@??2???EC?%?A???B?A??JA?dmA??o@?pA2?B??f?@?҅@?ՆC?	A?x?B ?AɌ?A???A҇?A???Af0RA܀UA?t?A???A?YA???@D#?@c˴A?|?B??A???A??tAxХA/N?AE?A??;A??A@?A? RAS??Ak?@?H?Am.A<jOA???A?D@z?&C??5A?cRAă3A??AA???AWV?Ah?B??@??????C?'?A??BeA?~?A???A?a?@???A3s??u?@???@?SC? A??B G            	      (   !   
      
   ?      	      !   2      ?      *   	                        F         h   7      8      B                           	      
   '      !      	   "   ?                        )            +            %   #   -   /      !                        #   1      !   )   !      '      '                        %                              )                        #                              -                                 #         !                  '                        #                                 N??TN???Oz?O?O??O?^P?CN*~?NH??Oa??O?P?N?ڃN6?|O???O~e5O?? O?k?O:?N?lOHSN???N'N_N??O??NΒ?N???N??O??N???O4>,OyɞO???O?	3O-?fOm?RN?N?P?eOVN?r?N?+|O'?}O2?5N7?O>?_O??NUU?O ՃN???O5??O<?Ol?O??NV??O??O???O?  ?  `  ?  Q  ?  ?  ?  ?    ?    ?  B  ?  ?  ?  q  ?  ?  ?  ?  M  ?  ?  9  ?  ?  k    ?  ?  
?  	h  ?  ?  b  
F  ?  u  ;  W  ?  0  Y  ?  ?  ?  d  j  ?  	Q  ?  ?  	?  K  {??9X??9X??o%   <t?;?`B<T??<D??<?o<??
=??T<?`B<???<?`B=#?
=C?<?`B=y?#<???=<j<???=o=o=+=+=0 ?=?w=,1=?1=,1=0 ?=?1=T??=T??=?C?=e`B=]/=]/=Y?=aG?=]/=q??=u=?7L=?o=?o=?o=?o=?C?=??-=??-=???=??
=???>+=?x?sprt???????tssssssss????????????????????!#/<FHOUMH?<7/'##????????????????????)57BCJHB=5)#zqpss|?????????????z?/;DW]^TC;/"z???????????zzzzzzzzBBINV[^^[NBBBBBBBBBBvuvwz?????????????zv????????????????????????????????????????????????????????????wwx??????????????|w??????????????????????????????????????????	)BN[g}}tgZ5?????????????????????????????????????????444<HUahnoqongaUQH<4????????????????????
)/*)







kekmnwz}zxsnkkkkkkkk' ")6BOOW[\][[OJB6)'????????????????????????????	?????????!*36CDHDC60*'#/<CQ\adWME?</&#?????
!#$##
?????????????????	)6BEF><6$????????????????????????????????????????????????????????????  )48::85/) #018840&#!?????)/:;;5)????????

####"
 ??		
#//33/,#
		"#$/<?EFD<9/-#""""UPRUX[^ht????tjhb[U???)/16860)$????

?????????????????

????????$)35BCLOPOMFB5$??????????????????????????????????????????$)-,)????????????????????????????????????????????SOV[ahttu?????{th[SS????????????????????????????????????????????? 
"###
????%"#)5BN[`ilkf[NB5,)%		
 #//97/#
	?zÇÓ×ÚÓÓÇ?z?y?u?y?z?z?z?z?z?z?z?z?{ŇŔŠŭŲŭŬŠŔŏŇ??{?v?y?{?{?{?{???????????????????????????????????????????????????????????????????????T?Z?`?g?m?n?m?a?`?T?G?@?;?8?9?;?@?G?O?T?O?[?h?tāčĚĝĤĠĚđā?t?[?U?I?A?D?O??"?;?T?a?f?o?p?m?a?T?H?)?	?????????	??a?m?s?z?~?{?z?y?m?a?`?]?a?a?a?a?a?a?a?a?n?{ŇōŇŃ?{?n?i?d?n?n?n?n?n?n?n?n?n?n?n?{ŇŔśŭŲŶųŠŜŏŇ??s?o?q?s?j?n????????????????ֺɺ????????ɺֺ??????!?-?-?0?6?-?!?????????????????/?<?C?>?<?/?#??#?%?/?/?/?/?/?/?/?/?/?/?\?h?uƁƈƌƉƅ?{?u?h?\?C?8?2?6?;?E?O?\??(?5?9?N?R?U?Z?N?A?5?(????????????????#????????????ſŻ?????????߿??Ŀݿ??????տ˿ȿ??????????????????????????????????????ݽнϽνԽݽ޽????????????u?s?n?m?s?v?????????(?5?A?N?U?U?S?N?D?A?5?(?&???????(?N?R?Z?`?a?`?Z?W?N?A?5?0?1?5?7???A?I?N?N?f?s?s?s?r?l?f?Z?Y?X?Z?c?f?f?f?f?f?f?f?f?a?n?zÇÈÇÅ?z?p?n?a?`?a?a?a?a?a?a?a?a?ʾ׾??????????????ؾ׾ʾ????????¾Ǿɾʿm?o?y???????????y?m?`?^?W?W?`?h?m?m?m?m?ûлԻܻ??ܻѻлû??????????????ûûûÿm?y?????????????????y?m?k?b?`?_?`?f?m?m?(?A?Z?j?y?|?s?f?Z?N?A?(??????????(?N?[?g?q?t?z?t?g?`?[?N?M?B?=???B?K?N?N?????????????|?y?s?l?j?`?[?Y?Y?_?`?l?y???:?S?x???x?p?_?[?U?N?J?F?:?-????!?-?:?????????ùϹ޹????ܹϹù??????????????ù??????????????????????éàÑÏÛãìù?H?U?[?a?l?i?`?U?H?<?/?#?#??#?+?2?<?C?H¦²????????²¦?z?t?p?s?~¦?ּ??????????????????ּμʼǼʼϼּּּ??#?0?<?I?N?S?R?K?>?0?
??????ĺ?????????#?????	???"?"????	??????????ܾ??????T?`?m?n?q?m?j?`?_?T?G?>?;?0?;?C?G?S?T?T?=?I?V?a?b?h?f?b?V?I?=?1?0?/?0?9?=?=?=?=?Y?f?r??????????????????????u?f?^?Y?M?Y?Y?e?r?~????????r?e?[?Y?T?L?B?@???L?P?YD?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D????	????"?"? ????	???????????????????????????$?+?+?$?????ƳƧƚƖƖƧ???
???????
???????????
?
?
?
?
?
?	???"?%?/?0?/?%?"???	???????????	??????????????????????????????????????伱?ʼּڼ߼ۼּӼʼļ???????????????????????????#?(?4?4?5?(????????????????3?@?L?Q?Y?^?e?e?e?e?Y?L?@?8?3?/?-?/?3?3?M?V?Y?`?Y?Q?M?@?4?'???
?
???'?4?@?M?????????????????x?l?f?l?x?y????????????E?E?E?E?E?E?E?E?E?E?E?E?E?E?EvEvE{E}E?E??O?[?h?wāĂ??y?t?h?[?O?B?7?3?5?;?B?G?O??*?6?C?F?O?W?O?M?C?B?*???????? 9 = $ B , ' $ m $ 7 !  V $    s  w  R Z U < 1 ; - = $ X ` ) . C 3  M 5 B  _   a b [ S 9 : E S F b X <  (  ?  ?     ?  <  2  d  ?  X  ?  ?    -  G  ?  ?  ?  ?  ?  ?  4  [  G  e  ?  ?  ?  ?    ?  A    ?  ?  ?  ?  ?  &  ?  ?  ?  ?  U  ?  Q  ?  *  ?  ?  ?  0  y  q  M    E  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  A[  ?  ?  ?  ?  ?  ?  ?  ?  ?  f  E     ?  ?  ?  E  ?  Y  ?  #  `  \  X  U  M  E  =  4  *        ?  ?  ?  ?  ?  ?  ?  ?  ?    N  u  ?  ?  ?  ?  e  A    ?  ?  b    ?    o  ?   ?  D  A  =  B  K  O  H  @  6  -  !      ?  ?  ?  ?  z     ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  x  Q  !  ?  ?  ^    ?  ?  ?  ?  l  M  5  %    ?  ?  n  +  ?  ?  $  ?  ?  ?   ?  ?  ?  ?  ?  ?  ?  ?  ?  l  7  ?  ?  c    ?  7  =    ?  U  ?  ?  ?  ?  ?  ?  ?  ?  ?  w  b  C    ?  ?  ?  S    ?  ?    ?  ?  ?  ?  ?  ?  ?  ?  }  m  \  L  2    ?  ?  ?  R   ?  ?  ?  ?  ?  b  r  ?  ?  ?  q  X  ?  (       !      ?  ?  F  ?  	?  
?  d  ?  g  ?        ?  v    q  
?  	^  &    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  q  R  ,    ?  ?  ?  7  ?  ?  B  F  J  F  <  1  $      ?  ?  ?  ?  ?  ?  q  T  7    ?  ?  ?  ?  ?  ?  ?  v  a  I  ,    ?  ?  n  2  ?  ?  I  ?  *     C  d  ?  ?  ?  ?  ?  ?  ?  z  Q    ?  ?  t  !  ?  J    ?  ?  ?  ?  ?  ?  ?  ?  ?    _  9  	  ?  {  !  ?  J  r  b  q  a  A  :  =  8  .  !    ?  ?  ?  ?  d    ?     ?  ?    9  Y  m  ~  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  _    ?  ?  ?  ~  ?  ?  z  t  n  g  ]  S  I  ?  7  /  '          ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  v  P    ?  ?  5  ?  6  h  O  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  q  R  3    ?  ?  ?  b  -   ?  M  G  A  <  6  0  *         ?  ?  ?  ?  ?  ?  ?  ?  t  `  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  }  u  l  _  O  :  "  	  ?  ?  ?  ?  ?  ?  ?  9  ,        ?  ?  ?  ?  ?  ?  ?  ?  ?  |  p  d  f  o  w  ~  ?  ?  ?  ?  ?  ?  ?  ?  s  V  .    ?  ?  l  +  ?  ?  ?  ?  {  f  O  7      ?  ?  ?  ?  ?  j  I  %  ?  ?  ?  H   ?  g  k  e  T  <       ?  ?  ?  R    ?  ?  (  ?  ?  g    ?  ?  ?  ?  ?  ?  ?    +  ]  ?  ?  ?     ?  ?  .  ?  ?    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  o  [  F  1    ?  ?  ?  ?  /   ?  ?  r  Z  A  %      ?  ?  ?  k  9    ?  ?  c    !  :  b  	E  	?  	?  
%  
o  
?  
?  
?  
?  
?  
?  
?  
p  
  	T  X  /  ?  5  %  	V  	c  	h  	c  	I  	'  ?  ?  y  2  ?  ?  D  ?  P  ?  ?  ?       s  ?  ?  ?  ?  h  >    ?  ?  w  ;     ?  ?  C  ?  ?  $  K     ?  *  $  C  c  {  ?  o  I    ?  f    ?  P  ?  ?  ?  I  .  B  P  \  a  _  U  C  .      ?  ?  ?  V  ?    ?  M   ?  
D  
D  
6  
  
  	?  	?  	O  ?  ?  Y    ?  0  ?     ?  ?  ;  ?  ?  ?  ?  ?  ?  i  J  +  	  ?  ?  ?  ?  f  D    ?  ?  ?    u  o  i  c  Y  G  1    ?  ?  ?  ?  p  G    ?  ?  P  ?  ?  )  6  3  &         ?  ?  ?  ?  ?  c  4    ?  ?  U    ?  W  G  5     ?  ?  ?  ?  L    ?  ?  h  C  .  ?  ?  ?  ?  U  ?  ?  ?  w  \  I  6       ?  ?  ?  k  3  ?  ?  _    ?  ?  0  (                 '  .  1       ?  ?  ?  f  ;    ?  ?  :  S  Y  G  *    ?  ?  Y    ?  ~  ;  ?  ?      ?  ?  ?  w  V  0    ?  ?  ?  ?  t  f  X  G  +  ?  ?  I  ?  )  ?  ?  ?  ?  ?  ?  u  a  N  ;  %    ?  ?  ?  ?  ?  h  F  %  ?  ?  ?  t  c  R  >  '       ?  ?  ?  ?  ?  ?  ?  ?  o  O  d  Z  P  B  3  &    
  ?  ?  ?  ?  ?  ?  c  =    ?  ?  ?  j  Z  I  A  E  ;  #  ?  ?  ?  M    ?  j    ?  f  +  ?  ?  ?  7  /  ,  '    	  ?  ?  ?  ?  ?  `  )  ?  f  ?  g  ?  O  	Q  	)  	  ?  ?  z  Q  !  ?  ?  q  3  ?  ?  A  ?  ?  ?  ?  m  ?  ?  ?  h  2  ?  ?  q  "  ?  ?  ?  o  6  ?  ?  	  m  ?    ?  ?  ?  ?  ?  ?  v  ]  B  '  
  ?  ?  ?  ?  V  *   ?   ?   ?  	?  	?  	?  	g  	>  	  ?  ?  d    ?  ?  -  ?  ?  	  w  ?  ?  Y  H  l  6  ?  g  ?  ?  #  B  J    ?  C  ?  ?  *  I  -  ?   W  {  V  -  ?  ?  ?  ?  o  ?  
  ?  ?  Y    ?  ?  2  ?  ?  G