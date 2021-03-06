CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?h?t?j~?   max       ??x????F      ?  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M??&   max       P??Q      ?  ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ?,1   max       >!??      ?  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @?0??
=q   max       @F?Q??     ?   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??=p??
    max       @vb?\(??     ?  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @$         max       @Q?           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @?'        max       @???          ?  2?   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ?ě?   max       >r?!      ?  3?   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A??"   max       B08      ?  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A???   max       B/?,      ?  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >?D^   max       C?{      ?  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >?<?   max       C?M      ?  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ?      ?  7?   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C      ?  8?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          '      ?  9?   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M??&   max       O??n      ?  :?   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ???E????   max       ???s??g?      ?  ;?   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ???   max       >!??      ?  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @?@        max       @F?Q??     ?  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ???Q??     max       @vb?\(??     ?  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @!         max       @Q?           t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @?'        max       @???          ?  O?   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A?   max         A?      ?  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ??e+??a   max       ???s??g?     ?  QX      
   -            Q            d   	   ?   4         b         D         	   )   "               M      9      S   T   
   +   &   .                     ?            "      	         
   Q   O??CN)L8O̸?N??6N?LN?P??QM???M??&N?"2P??7O?P?F?O?*ORg?O?<P^-2O3,?N7	O??hN?n?N??N@??O?O??Oe ?O4?7N??N[??O?MoOBy?O?j9N?~?O??!P*fNw8O̅O???O?/Nl?3O?8oN?NN4?lN??YO?pOԔ?O?+O??N??HO??N???N??N??N???NS{4O;??N????,1?+??j?D?????
???
;D??;D??;??
<o<e`B<e`B<u<?t?<?t?<?1<?1<?9X<?j<ě?<ě?<???<???<?`B<?=+=C?=?w=#?
='??=,1=0 ?=8Q?=8Q?=<j=@?=D??=D??=P?`=P?`=T??=Y?=e`B=e`B=m?h=q??=y?#=?%=?o=?o=??w=?^5=???=???=?l?=???>!??WVWZgt????????|tgc[W????????????????????RUW^gt??????????tgaR????????????????????????????????????????#020&#??????)ZbtpaB)??????????????????????????	?????????????????????????????????????)BN[jne[B)??)05BGNNNSNB65)()9g??????tg[C5,,)????
/39<<7/'#
??{v?????????????????{757;BHTacdbaYTLH@;77??????)BHJJ?5)????)5BKLNSB>5)	?&))6?BHMB>6)&&&&&&&&???????????????????? */6CHGC965*"????????????????????????????????????????".6BOY]^__][OB:6)jnv???????????????tjXU[_dht??????????thXhfhktv????????????th????????????????????)),16BGHFB6)))))))))????"/31#??????{{?????????????????{ 
#-9?JJE</!?????? ??????????
#/<HNSURH</#xvz???????????????x???????????????????????
 %.4<=91/#????? 
#/04:973*#
??)??????/9;86-)????????????????????$5BNgsz}wg[N5*$!*,/4<@HPMHB<1/******??????????????hhort???????ztlhhhhhdtmihmz???????????md????????

????/69:<DHT_aa]YTQKH;//?????????????????????????????????????????????????! ???

#*.*#








????????????????????mnrvz???????zmmmmmmm#/5<?<</-#????????????????????nz??????????zunlgghn??????????zutwz?????a?n?zÇÞæãÓÇ?z?p?a?U?K?H?>?B?H?O?a??#?)?+?#??
?
?
?????????????????????????????????????????????????????????????????????????????????????????čęĚĦīħĦĚčā?~āăĄčččččč??'?/?2?2?'?#????????????????ѿ?????/?J?_???A????ѿ????????????;?A?G?S?G?;?.?"? ?"?.?:?;?;?;?;?;?;?;?;??????????ùõøù?????????????????????ſm?y?y?|?y?x?n?m?`?T?S?J?T?X?`?c?m?m?m?mƳ?????????#??	????ƧƁ?u?j?[?^?xƎƳ?;?G?T?\?`?b?a?`?T?G?F?;?8?9?.?.?.?.?:?;āčĝĦĕĄ?|?t?[?6??????????6?h?vā?m?y?????????????y?m?`?T?E?;?/?;???Q?i?m???%?'?'?????????????????????????/?;?H?P?T?Z?\?Y?T?H?;?8?/?*?"??"?&?/?/???
?U?a?e?^?[?R?F?5??
??????ĽĹ???????f?s?v?v?x?{?v?s?f?`?Z?P?M?H?I?L?T?Z?d?f?ѿݿݿ????????ݿֿѿʿɿѿѿѿѿѿѿѿѹ??Ϲܹ??????????ù??????????????????????????????????????y?m?e?`?_?`?g?m?y?|???	?	?????????
??????	?	?	?	?	?	?	?	???????????????????????????????????????Ҿ??ʾ׾????????׾ʾ???????{?|???????????????????????????????s?m?j?X?_?]?^?a??-?:?@?F?_?e?`?J?F?:?-?!????????-??#?+?/?4?<?D?G?B?<?:?/?#? ??
?????A?M?R?T?X?M?A?@?:?A?A?A?A?A?A?A?A?A?A?A?????????????????????????????????????@?M?c?m?n?i?f?Y?M?@?'????? ?'?4?>?@?????????????ĺ??????????????~?s?y?~???????ʾ???????????ʾ????????y?s?k?s????5?B?N?[?d?c?[?N?B?5?,?/?5?5?5?5?5?5?5?5EE*E-E5E:E8E0E$EEED?D?D?D?D?D?EEEàùþ??????ÿüíÓÇ?w?g?_?\?n??zÇà???????????????????ݻ??????????????T?o?z?x?l?a?T?;?/??	???????????	?"?;?T???????????????????????????????x?u?|?????<?<???I?L?Y?e?r?????ɺӺѺɺ??????~?e?<E?E?E?E?E?E?E?E}EvEzE?E?E?E?E?E?E?E?E?E??T?a?m?m?h?^?;?/?"??	??????? ?	??/?H?T????? ????????????????????????????????m?z?????????z?m?a?]?a?h?m?m?m?m?m?m?m?m?4?4?@?M?S?X?M?B?@?4?3?)?'?%?'?3?4?4?4?4ŔŠŨŹ????????????????ŹŝřššŞŐŔDoD{D?D?D?D?D?D?D?D?D?D?D?D?D{DkDdDcDgDo?0?<?I?U?b?n?vŃŇŇ?n?b?I???<?0?,?,?.?0???ݽ????ݽнǽƽ?????????????????????????"?'?*?-?"???	?	?	?	???????????????????	???"???	?????????????????????????????????????????????????????????:?F?S?_?l?x?????x?l?_?S?F?:?-?!?-?1?:?:??????#??????????????????????*?6?@?>?7?6?4?*?????
????¿????????¿²²«²³»¿¿¿¿¿¿¿¿???????????ܻл˻Ȼ̻лٻܻ??????:?C?:?.?!???????????????!?.?:?:?: < 0  T ( k D ? ? P $ 8 6 2 q - F V g 6 P ` N C L C 6 G 0 > ) W H . B 1 f ; K T F ( f 4 V  w g R 5 ( j : 5 z " y    	  F  ?  ?  ?  L    E  T  ?  A  3  ?  0    "  ?  ?  v  ?  ?  O  z  ^  r  ?  ?  ?  s  ?  ?  ?  ?    2  ?  =  ,  g  ?  [  ?  P  ?  -  ?  ?  -  ?  ?  ?  *  ?  %  ?    ?T???ě?<?1??`B?o?o=?1;ě?<#?
<T??=?x?<?j>["?=?O?=,1=\)=???=\)<???=?j<??h<?=\)=?O?=?%=<j=q??=,1=<j=???=?t?=???=e`B>J>?=ix?=?Q?=?{=Ƨ?=y?#=???=?O?=q??=?O?=??T>r?!=??P=??
=?O?=Ƨ?=?^5=???=?`B=???=??m>Q??>1&?B	??BB	?EB?B ?GB%1B?BL?B9\B?UB??B=?B	YvBfB??A??B?3B?CB?eB"WB08B-_B??BN?B?RB?B8?B"??B݅B?oB؋B|?BB3?B?B"6aBnBwDB}?B?BNB??B#"B??BDHBSA??"B+??B;?B?kB$??B--Bx?B?BReB??B??B	??B2?B	??BI;B zB$?*B@B>sB?B??B??B=?B	@4BH{B?A???B?SB??B?vB??B/?,B2[B?BB5B??BB!BB?B"??B?AB??B??B#_B??B?B?\B"@B?tBH?B??B ?B=?B??B.?B?9B\_B>"A???B,8?B?ZB??B$?B-D"B?$BB?B??B?pBAfA?A??OA?jAA??A???@??EA?xAb?JA?m?AiƮBC?Ae?A?=?Ak.	?^?(A??
A?{?A@??A|b[>?D^An<?A?e?A??AM?AE??@x]8A?"5A<lz@R?@??v@I?AOGA?C?iA?7,@??A?yEA???@	s?C?{A??vA??'A?>?@н?A?AC???A??NA$??A???A???AJ??@???A2b,A??A?c?@??|A?SAȂ?A??VA??A???Aރ[@?3?A?? Ad??A?s?Ai??B??Ae?A?{cAk&H?S?SA?d?A???A???A{{>?<?Am
A???A?\?AM2AF=?@|?<A?}?A<@L??@?@?	AJ??A??C?o?A??@?VwA?sbA??@k?C?MA??rA?c?A?x?@??A??C??A??HA$??A??7A??AJ?n@?YBA3A?u{A?}?@???A?      
   -            R            e   
   ?   4         b         E         
   *   "               N      9      S   T      +   &   /                     ?            "      
            R                        C            5      7            /                        %                     '         -      '      %      %            %                                                                     !                  %                                             #               '      %      #            !                                    OO&?N)L8O[*N??6N?LN?O??BM???M??&N?"2O?6fO?OnǭOB??N?=mO?<O??nO)?N7	OI?N?n?N??N?|ODg?O?k?Oe ?N??aN??N[??Oz:?OBy?O??N?~?Oc?O???Nw8O̅O]??O?/Nl?3O??=N>??N4?lN??YO??0O+'mOaO??N??HO?N???N??N??N???NS{4O8*?N???  ?  ?  ?  ?  t  {  ?  ?    E  @  ?    	?  ?  ?  
|  A    	}  ?  ?  X  ?  7  K  ?      
?  <  	?  [  ?  
?    ?  ?  ?  ?  ?  Z  ?  3  ?    ?    ?  t  .  ?  ?  4  ?  3  ~????+?#?
?D?????
???
=D??;D??;??
<o=aG?<e`B>$?<??h<?/<?1=T??<ě?<?j=49X<ě?<???<?/=??=#?
=+=,1=?w=#?
=u=,1=@?=8Q?=]/=??w=@?=D??=Y?=P?`=P?`=Y?=ix?=e`B=e`B=q??>
=q=}??=?%=?o=?O?=??w=?^5=???=???=?l?>   >!??ZXY[]cgty??????tg^[Z????????????????????Z\`got??????????tg[Z????????????????????????????????????????#020&#&)5BNSYZUOB5)???????????????????????	???????????????????????????????????)5BJPWWUNB)?)05BGNNNSNB65)(KIJNP[gt|?????ytg[PK??	
#//577/&#
 ??????????????????757;BHTacdbaYTLH@;77?????4<?=5)????)5?BILB95)"&))6?BHMB>6)&&&&&&&&???????????????????? */6CHGC965*"????????????????????????????????????????)).6BOSX[\\[TOB@6,))yy?????????????????yXU[_dht??????????thXttz~????????????|ttt????????????????????)),16BGHFB6)))))))))?????$(&"????{{?????????????????{	
#+7>EHHC</#	?????? ??????????
#/<HKPRNH</#???????????????????????????????????????????
 %.4<=91/#?????#/47640/'#
)??????/9;86-)????????????????????)5BNfry|vg[N5+%"-./<HMJH=<:/--------??????????????hhort???????ztlhhhhhjhmz?????????????zmj?????????

?????79;;=EHNT_`][XTPIH;7??????????????????????????????????????????????????

#*.*#








????????????????????mnrvz???????zmmmmmmm#(/,'#????????????????????mhginz??????????zunm??????????zutwz?????a?n?zÁÇÓÚáàÜÓÇ?z?a?T?J?M?U?[?a??#?)?+?#??
?
?
???????????????????????????????????????????????????????????????????????????????????????????čęĚĦīħĦĚčā?~āăĄčččččč??'?/?2?2?'?#??????????????ݿ?????(?+?0?3?*????????޿ؿտۿݿ;?A?G?S?G?;?.?"? ?"?.?:?;?;?;?;?;?;?;?;??????????ùõøù?????????????????????ſm?y?y?|?y?x?n?m?`?T?S?J?T?X?`?c?m?m?m?mƳ??????????????????ƳƧƚƏƆ?|?}ƈƚƳ?;?G?T?\?`?b?a?`?T?G?F?;?8?9?.?.?.?.?:?;?B?O?[?e?h?o?p?m?h?[?O?B?6?0?)?&?$?*?6?B?y???????????????y?m?`?P?G?C?G?J?T?Z?m?y??????????????????????????/?;?H?P?T?Z?\?Y?T?H?;?8?/?*?"??"?&?/?/??#?0?C?J?F?>?3??
????????????????????Z?f?s?s?s?t?w?s?q?f?Z?T?M?K?K?M?O?V?Z?Z?ѿݿݿ????????ݿֿѿʿɿѿѿѿѿѿѿѿѹùϹܹ??????????????ܹϹ??????????????ÿ??????????????????y?m?e?`?_?`?g?m?y?|???	?	?????????
??????	?	?	?	?	?	?	?	???????????????????????????????????????Ҿ??ʾ׾޾??ݾ׾ʾ????????????????????????????????????????????s?f?d?c?f?e?h?s??-?:?@?F?_?e?`?J?F?:?-?!????????-?#?#?/?<?>?A?>?<?2?/?+?#??????"?#?#?A?M?R?T?X?M?A?@?:?A?A?A?A?A?A?A?A?A?A?A?????????????????????????????????????@?M?Y?_?d?g?e?`?Y?M?@?4?-?'??? ?'?4?@?????????????ĺ??????????????~?s?y?~???????ʾ??????	?????ʾ??????????{?u?t????5?B?N?[?d?c?[?N?B?5?,?/?5?5?5?5?5?5?5?5EEEE*E1E7E5E,E!EEED?D?D?D?D?D?D?EÓàìñôòïìêàÓÇ?z?v?n?k?l?o?zÓ???????????????????ݻ??????????????T?o?z?x?l?a?T?;?/??	???????????	?"?;?T?????????????????????????????????|?y????<?<???I?L?Y?e?r?????ɺӺѺɺ??????~?e?<E?E?E?E?E?E?E?E}EvEzE?E?E?E?E?E?E?E?E?E??T?a?m?k?f?]?;?/?"??	????????	??/?H?T????????????????????????????????????????m?z?????????z?m?a?]?a?h?m?m?m?m?m?m?m?m?4?4?@?M?S?X?M?B?@?4?3?)?'?%?'?3?4?4?4?4Ź????????????????????ŹŭşŜŢŢŜūŹD?D?D?D?D?D?D?D?D?D?D?D?D?D?D{DtDuD{D?D??0?<?I?U?b?n?u?{Ł?{?n?b?U?I?@?<?0?.?.?0???ݽ????ݽнǽƽ?????????????????????????"?'?*?-?"???	?	?	?	?????????????????????	????	? ?????????????????????????????????????????????????????????:?F?S?_?l?x?????x?l?_?S?F?:?-?!?-?1?:?:??????#?????????????????????*?6?:?:?6?*????????????¿????????¿²²«²³»¿¿¿¿¿¿¿¿?ܻ???????????????ܻл˻Ȼ̻лڻܽ:?C?:?.?!???????????????!?.?:?:?: 1 0  T ( k # ? ? P  8  * L - 9 V g , P ` 9 < 0 C C G 0 + ) Z H -  1 f 0 K T C ( f 4 W  n g R 9 ( j : 0 z " y    ?  F  ?  ?  ?  L  0  E  T  ?  ?  3  ?  ?  ?  "  O  N  v  ?  ?  O  J  ?  '  ?  ?  ?  s  ?  ?  s  ?  ?  ?  ?  =  ?  g  ?  8  Y  P  ?    f  h  -  ?  ?  ?  *  ?  ?  ?  y    A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  A?  |  ?  ?  ?  ?  ?  ?  ?  ~  b  D  (        
  ?  ?  q  *  ?  ?  |  u  m  e  ]  U  L  C  9  0  &              ?  V  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ]  5    ?  ]  ?  8  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  x  k  _  Z  \  ]  ^  V  K  ?  4  t  p  k  f  b  ]  Y  U  R  O  L  I  F  P  s  ?  ?  ?  ?     {  o  b  V  J  =  1      ?  ?  ?  ?  ?  i  G  %     ?   ?  ?  I  ?  )  g  ?  ?    S  ?  ?  ?  ?  P  ?  @  ?  ?  ?   |  ?  ?      ,  <  L  V  ^  f  n  v  }  ?  ?  ?  ?  ?  ?  ?         ?  ?  ?  ?  ?  ?  ?    5  G  M  S  Y  m  ?  ?  ?  E  C  B  A  ?  :  6  1  ,  '  "              ?  ?  ?  ?  ?  X  ?  ?    0  =  @  4    ?  ?  U    ?  ?  *    A  ?  ?  ?  ?  ?  ?  ?  ~  f  M  3       ?  ?  ?  d     ?   ?  
F  ?  q    ?  	  ?  ?  \  ?  ?    ?  ?      ?  
?  p  ?  	'  	\  	?  	?  	?  	?  	~  	Q  	  ?  ?  B  ?  M  ?  "  ?  ?  ?  M  ?    R  ~  ?  ?  ?  ?  ?  ?  ?  m  T  5    ?  ?  ?  ?  w  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  o  J      ?  ?  u  #  ?  d  	  	?  	?  
G  
n  
x  
{  
q  
P  
   	?  	w  	  ?  +  ?    I  #  4  0  9  ?  A  <  5  ,        ?  ?  ?  ?  ?  ?  ?  ?  ?  ?      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ^  ?  	  	L  	k  	|  	{  	j  	M  	'  ?  ?  ?  <  ?  1  t  x  ?  (  ?  v  l  a  W  M  C  8  ,       ?   ?   ?   ?   ?   ?   ?   ?   ?  ?  ?  x  c  R  N  J  F  A  7  .  %    
  ?  ?  ?  ?  ?  ?  R  U  W  M  A  /    	  ?  ?  ?  ?  v  P  +    ?  ?  w  :  9  Y  ?  ?  ?  ?  ?  y  W  1    ?  ?  d    ?  3  ?  ?  ?  ?  ?  ?      3  6  *    ?  ?  ?  W  "  ?  ?  1  ?  5  ?  K    ?  ?  ?  ?  X  *  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  Q  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  O    ?  2  ?  ]  ?  |            ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  {  _  9    ?  ?  ?  ?  	?  
J  
?  
?  
?  
?  
?  
?  
?  
`  
$  	?  	o  ?  T  ?  ?  ?  ?  t  <    ?  ?  ?  s  Y  A  &    ?  ?  g    ?  ?  ?     ?  K  	r  	?  	?  	u  	_  	C  	!  ?  ?  w  )  ?  U  ?  ?  9  ?  ?  ?  \  [  S  J  ?  3  &      ?  ?  ?  ?  ?  ?  z  b  M  <  +    ?  ?  ?  ?  ?  ?  ?  n  3  ?  ?    ?  ?  B  
?  	?  	7  p  ?  f  ?  	  	F  
  
d  
?  
?  
?  
?  
{  
<  	?  	?  ?  Q  ]    ?  5    
  ?  ?  ?  ?  ?  ?  k  S  <  '      ?  ?  ?  ?  ?  D  ?  ?  ?  ?  v  [  8    ?  ?  z  &  ?  s    |  a    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  m  H    ?  ?  E  ?      ?  U  ?  ?  r  A    ?  ?  l  E    ?  ?  V     ?    ?    ?  ?  ?  ?  ?  ?  ?  j  L  .    ?  ?  ?  ~  ]  ?  !      ,  ]  ?  ?  ?  ?  ?  ?  ?  ?  d  7  ?  ?  Y  u  d  L    ?  6  ?    +  ?  N  W  X  R  B  .    ?  ?  ?  ?  ^  5  
  ?  ?  ?  ?  ?  |  s  j  a  X  U  U  U  U  U  U  X  b  l  u    ?  ?  3  ,    ?  ?  ?  ?  u  Q  ,    ?  ?  s  O  0    ?  /  ?  ?  ?  ?  ?  ?  t  c  P  <  !  ?  ?  ?  n  2  ?  ?  2  ?      >  V  '  ?  b  ?  ?    ?  ?  ~  ?  ?  ?  E  z  ?  ?  	?  {  ?  ?  ?  ?  y  h  T  C  3  %    ?  ?  ?  ?  u  ?  `  ?      ?  ?  ?  ?  ?  ?  }  a  @    ?  ?  ?  ?  T    ?    ?  ?  ?  ?  ?  }  n  _  P  @  0           ?  ?  ?  ?  y  A  k  p  t  r  o  f  V  @  %    ?  ?  ?  J  ?  ?  I  ?  ?  .    ?  ?  ?  ?  Y  &  ?  ?  ?  `  1  ?  ?  v  (  ?  ?  2  ?  ?  |  i  V  A  ,    ?  ?  ?  ?  ?  ?    j  Q  5     ?  ?  t  ^  H  3    ?  ?  ?  ?  ?  }  j  ^  a  g  o  z  ?  ?    ?  ?  ?  )  4  (      ?  ?  ?  ?  ?  l  R  ;  =  ?  ?  ?  u  B    ?  ?  ?  ?  ?  ?  ?  ?  ?  w  i  _  V  M  E  =  -  /  "  ?  ?  ?  R    ?  g    ?  
  `  ?  
?  	  +    ?  ~  b  =        ?  ?  ?  \  5    ?  ?  ?  |  M  ?    _