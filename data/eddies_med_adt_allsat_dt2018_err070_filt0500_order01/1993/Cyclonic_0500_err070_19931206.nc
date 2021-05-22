CDF       
      obs    J   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�n��O�<     (  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P���     (  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =�o     (  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�Q�   max       @FS33333     �  !$   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @v
=p��     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @Q�           �  8D   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�)�       max       @��@         (  8�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��`B   max       =u     (  :    latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��d   max       B0Н     (  ;(   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�e-   max       B0��     (  <P   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >Q��   max       C��R     (  =x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�q   max       C���     (  >�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          I     (  ?�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?     (  @�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =     (  B   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P���     (  C@   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�h   max       ?��m\���     (  Dh   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���
   max       =�o     (  E�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�Q�   max       @F<(�\     �  F�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @v|z�G�     �  RH   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @Q�           �  ]�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�r        max       @��@         (  ^l   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�     (  _�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?vOv_خ   max       ?��m\���        `�         +            
   4   
      
      &         D      $      %               @   8   :            E               <      H         	   	   	   	                                        ;      -            !                                 N݋NUfO���N{�O��OE��Nf�P���Nf�O6�N<֑NR�O�X(N���Nm/�P��NO
O��wN�1�PN1.O�AN S"O�G�O�>�P%��P���Poo�NN�O�O�R�Pg�OxB�N��nPݑNÈCO�0%O�#P0N�$N��kN���N�vTO
�@N��aO�u O.��N��O��$NŬ<M���O��OsrO0�N"Ox�,OMu�O�D�N_ZO��5N턶O���NǅVO�1O�T-O�\�O�'�N�9nNO�NE<WO�N���O�NkGDN�*�=�o=T��<���<u<49X��o�o��o���
�#�
�#�
�D����o��C���C���C����㼛�㼣�
��1��1��9X��9X�ě����ͼ�����/��`B��`B��`B��h�����������o�o�+�+�+�C��\)�\)�#�
�0 Ž0 Ž<j�D���H�9�L�ͽL�ͽL�ͽY��]/�]/�]/�]/�]/�aG��aG��}󶽃o�����7L��7L��O߽�\)��hs��hs����������������)56>6)������������������������������������������������������������������������������������!&(��������
!
���������HZ^Yaz����������zaMHrt�������tnnrrrrrrrr�����������������������������MNU[agojgb_[NLMMMMMM
#<UZ[ZZUM</#
��������������������#-/45/'#"$HUj�������zaU<'./;;?C>;/.*(........����)/���������?IRUVbjngbUNIC??????Tm�������������zmWPT���������
#+/2/#
Wht�����������thZVVW�������0,)������y���������������{ury��
#0Ib{�����zI0���@V\Z]m����������t[B@#&/<<?<6/#o����������������to#+6COZ\dijqg\RC6-*##��������27*�������������� ���������.5?BKNZVVNB54/......[g���������������tl[���


���������%6GQ[hhp}�����th[B)%Vam�������vsqmdaWTTV��������������������JO[hprph[[OMJJJJJJJJ�������������������������������������������������������� #)/<=ABB<2/#!;<EHUaaagda^UPH<<;;;����������������||}�������(�����-6BOXROB86----------��������������������$).6BBCBA=;6)! pt���������tpppppppp��������������������stv����������tqhijns����������������������������������������KN[mt�������tg[TMIKK��������������������#/<GX\adgd`UH</##inyz{�}{zrnkjiiiiiinvz|����������ndbdhn	
#$18:430)#
	#IUhn{����}nU<0!UPHH=</+(),/5<DHUUUU<BFQ[g�������gPGDE<<)5N[cgc[NB50#����������������������!!�����������������������������������������������"#08<==<0'#!""""""""05:<>IJMORSQLI<20//0�������������|������{�����������ztnmnoz{�����������������

#$$#
	�ֺӺѺֺغ�����ֺֺֺֺֺֺֺֺֺ�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��
�����������������
�#�/�4�7�9�7�/�#��
����������������������������������������������������*�6�C�O�\�g�e�\�H�6����¦¦²¿����������¿²­¦�����������������¼¼���������������������ĿĦč�~ăČģĩĞĤĿ������,�$��
���H�F�?�H�K�U�[�a�f�g�a�U�H�H�H�H�H�H�H�H�O�N�H�H�O�\�h�uƁƉƎƚƚƔƎƁ�u�h�\�O�ּϼʼ��ʼּ������ּּּּּּּֿ����������������������������������������)�!�&�$�&�$�)�6�B�O�[�h�s�o�Z�O�B�6�.�)������������������������������H�F�<�3�<�>�H�U�_�a�c�a�U�L�H�H�H�H�H�H�M�A�0�.�4�M�Z�s���������������s�f�Z�M�/�-�/�/�;�H�T�W�U�T�H�;�/�/�/�/�/�/�/�/�n�_�`�g�ŔŠ��������������ŵŭŠŇ�{�n�ʼ��������żʼּ����߼ּʼʼʼʼʼ���ƳƹƷ������������$�0�5�3�,���������������������	���"�*�+�"��	����ììàßÝàçìîöõíìììììììì�����{������������нݽ��������нĽ���àÙÏÇ�z�o�f�c�l�n�z�~ÅØìúüóìà�������y�q�p�x���������Ŀ��� �&���Ŀ����������k�a�Q�E�<�G�g�s�������������������²�[�=�7�N�g¦°��������������0�)�$�!�#�$�0�8�=�@�>�=�0�0�0�0�0�0�0�0��}�r�p�h�m�r������������������������ʾ��������������ʾ׾�����������׾ʼ����r�r���������ɽ�!�.�9�;�6�%���ּ������������~�v�w�y���������ſʿѿ�ݿѿ�������������������������������������������������������������������'�(������²¬ª²´¼¿����������������¿²²²²��ܹù����������ùϹܺ��,�0�,�$������5��������-�A�N�g�s�������������s�5�l�S�?�8�<�D�Y�l�������Ȼλڻܻл����x�l���������������������������������������������������ʾ׾����׾ʾ��������������s�m�i�j�o�s�����������������������s�s�"������"�#�/�2�;�B�E�;�/�$�"�"�"�"�/�'�"��	��	����"�/�;�C�H�I�I�H�;�/���������������������������������������Ŕ�|�{�t�{ŇŋŠŹ����������������ŹŭŔ��w�r�j�f�d�d�f�k�r��������������������}�w�u����������������������I�0�$������������
��#�1�<�K�Q�R�L�I�ѿϿĿ����������Ŀȿѿݿ����ݿԿѿѺr�o�n�o�r�}�~�����~�{�x�r�r�r�r�r�r�r�r�����������Ǻɺֺ�����������ֺɺ������������������	��"�*�0�3�2�/�"��	���𺽺��������ǺɺϺֺ����������ٺɺ��x�x�m�l�l�l�x�}���������x�x�x�x�x�x�x�x�׾;̾׾پ����	�������	�����׿	��������������	���$�!����	E�E�E�E�E�FFF$F=FJFJF?F;F;F6F'FE�E�E޹������������ùϹܹ�ܹٹϹù��������������������'�@�L�e�r�~�����~�]�L�@�'����l�j�_�S�F�C�E�F�S�_�l�x���������y�x�m�l�������������ƻл��"������лû���D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��h�g�tāĈĉĚĦĳĿ��������ĿĦĚč�t�hÓÇ�r�m�sÆÍÓàìù����ýùìçáàÓ�P�C�:�7�:�G�S�l�����������������y�l�`�P�ʽĽ������Ľҽݽ����(�#�������ݽʿ������������(�)�3�/�(� ��������������	������������������������򽫽����������ĽƽϽĽ�������������������� �����ݽѽݽ������(�-�3�(�!��� ÓÒÊÇÁÇÓàìïñíìàÓÓÓÓÓÓ�������������ĿѿݿݿܿٿտѿǿĿ�������D�D�D�D�D�D�EEEEEED�D�D�D�D�D�D�D�EEEE(E*E7E@ECEPESE\E`E]E\EPECE7E*EE J ` 0 n J - : / 3 : D u D P * Q B h J 6 = m * t G [ P O L  d i ) - . > � - G A ; ( % P @ J z R E � M + V x H ; K Z b 5 X 8 i D ; C J V 9 ] 9 4 Q W  K  �  ]  �  �  �  w  
  n  �  W  |  _  �  s  �  w  �  �  l  (  �  �  �      $  {  A  �  m  �  �  �  �  B  	  �  �  �  .  �  1  �  6  �  �  l  �  �  B  �  �  ^    �  �  �  O    �  �  �  R  3  s    �  Z  D  �  9  �  �=u=8Q켋C�<t����
��`B�49X�e`B�e`B��1���㼃o�Y��ě���h��{���ͽaG����m�h�\)���ͽP�`�0 Ž�9X���1�t��49X�]/�ȴ9�'�P�m�h�]/��Q�y�#������w��P�'0 Ž0 Ž49X�q����hs�L�ͽ���Y��T����C����o�u������C���`B�m�h�ȴ9�}󶽟�w����ě���9X����ě��� Ž�\)������1��j�ȴ9�Ƨ��G�BD�B�By>B �;B Bj�B$e>B daB��BF�B��B��B��B;�B�vB��A��dB�7B'7@B �BCZBtvB��BX�B*v�B&y6B
�sB2qB�B0НB,�hB��B�B{�BDBq�A��4B &BXQB |B!8B�=B�B��B7xB�B.2B�B�tBܡB,9BgB!�+B ̫B	{�B#�B��B DB�B%5B&�*B�8B	s�BpgB�#B��Ba&B6FB%�LB&u�B
�>B��BѮB�B=YB<�BdHB ��B�B@�B$E�A���B�BB|B�,B��B�hB:�B�XB��A�e-B�?B'$�B �DB?�BOB��B?�B*M�B&/B
��BA�B�B0��B-.�B��B��BD1B@BU�A��1B@CB��B�sB!��BH�B�B� B?�B��B�B>�B��B��B>vB>B!B
B ªB	i>B6�BIAB=fB<�B%I�B&�B�NB	A�BϽB�7B��B�B;�B%��B&b�B
ųB��B��BDW@At�C�>�A�оAЗLA�t�A���@���A��AŇ�B��A�sAt7A��UAӘcA��TAA(^A��A��A B&B�AZ�A̗TA%�HA��Ay?�A�0`A���B
A�@��AS'�A�%AtʢA�?�A�B�A��/?"AA��(@�0A��AP�+AE@$A��WA�H+A��YA�ʗ@��@���A���A{ �?���@;�A��;@@O�@���AW�nAY�]C��R>Q��?���@��@��C��+A�A�� A:�A/ �A���B��A%�A1]A� �AyC�N"C��@?�C�IA��AϦRA�ߚA� �@�ˊA�>A�uoB�AK�At�A�0�A��AĀ A>�vA�xSA�}�@�5/B��A[ �Ã3A#N�AɁAx�A�f�A��B
?b@�Q�AS!<A
As�A��9A��IA�~�?-�GA���@���A�KAAO�AE�A���A�sA�Q�A��@���@��A��Az�?�W@@b�A��a@4n�@�qAX2NAX�nC���>�q?���@�@�՛C���A��A˅�A��A0�'A�DB��A$�
A3�_A�gAx��C�HC��a         +            
   5   
      
      &         E      %      &               @   9   :   	         F               <      I         	   
   	   
                        !               ;      -            !                                                          7                        +      +      -         !   !   -   =   =            ?         '      '   +   -                                                         #      %      +      #         !                                                -                        %      #      -            !   +   =   !            3               !   #   %                                                                                                            N݋NUfOP%N{�OJ��OE��Nf�PGҬNf�O6�N<֑NR�O$K�N��0NK�O�NO
O��_N�1�PN1.O�AN S"Ow��O�>�P�iP���O�ʣN >N��uO ��PFM�OxB�N��nN�@N1I$O��%O��P��N�$N��kN���N�vTN�oDN��Os�aO	'�N��OP�NŬ<M���O��Oi&�O0�N"OW�OMu�O�m9N_ZO�]�N�<{O��NǅVO��rO�T-O�\�O~�sNЇ�NO�NE<WO�N���O�NkGDN��#  +  :    ^  [  �  �  B  �  O  �  U  %  �  �  �  G  �    �  �  �    �  �  �  [  �  0  �  �  �  V  �  �  �  �  �  �    n  <  �  �  �  S  �  �  m  =  �  p  �    �  �  
�  �  �  A  �  �  C  �    �  �  	5  y  &    �  	9  u=�o=T��<u<u;��
��o�o�D�����
�#�
�#�
�D����/��t���t���������������
��1��1��9X���ě��\)�����T����h�o�\)�t������D����P�0 Ž\)�0 Ž+�+�+�C��t��t��'@��0 ŽL�ͽD���H�9�L�ͽP�`�L�ͽY��e`B�]/�m�h�]/�}�e`B�y�#�}󶽉7L�����7L��\)��\)��\)��hs��hs���������������
)56>6)������������������������������������������������������������������������������������!&(��������
!
���������Wgdmz����������aTMNWrt�������tnnrrrrrrrr�����������������������������MNU[agojgb_[NLMMMMMM"#/<HPQQMHB<2/*#"��������������������#+/34/#)HUc�����|naWH</ ./;;?C>;/.*(........�������������?IRUVbjngbUNIC??????Tm�������������zmWPT���������
#+/2/#
cht����������thca``c�������0,)������y���������������~zwy��
#0Ib{�����zI0���fit�����������tkkgff#(/:<?<4/#!��������������������36CO[__^\UOEC96/,*+3������/21#�������������� ���������.5?BKNZVVNB54/......������������������������

�����������57BFO[hu{~��~~th[QB5am������tqpmaWUVWXa��������������������JO[hprph[[OMJJJJJJJJ��������������������������������������������������������#/;<?@A<0/#"GHUafba]ULHB>?GGGGGG~����������������}}~����������-6BOXROB86----------��������������������$).6BBCBA=;6)! pt���������tpppppppp��������������������stx����������triikns����������������������������������������MN[ht�������tg[VNKMM��������������������#/<DPVY^a`UH</##inyz{�}{zrnkjiiiiiilnz�����������zngegl#/07800# *0<IUZbnrvqncUF<0%%*UPHH=</+(),/5<DHUUUUN[g���������tg_RIGHN)5N[cgc[NB50#������������������������������������������������������������������"#08<==<0'#!""""""""05:<>IJMORSQLI<20//0�������������|������{�����������ztnmnoz{�����������������
##$#
	�ֺӺѺֺغ�����ֺֺֺֺֺֺֺֺֺ�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���
�����������������
��#�,�0�1�/�#���������������������������������������������������������*�6�C�I�J�C�6�,���¦¦²¿����������¿²­¦�����������������¼¼���������������������ĦĚĎďĚĦıĮĺĿ����$���
�������H�F�?�H�K�U�[�a�f�g�a�U�H�H�H�H�H�H�H�H�O�N�H�H�O�\�h�uƁƉƎƚƚƔƎƁ�u�h�\�O�ּϼʼ��ʼּ������ּּּּּּּֿ����������������������������������������)�)�)�+�.�6�B�O�[�a�h�l�h�f�[�R�O�B�6�)�������������������������������H�H�<�5�<�@�H�U�]�`�U�H�H�H�H�H�H�H�H�H�M�A�3�1�4�M�Z�s���������������z�s�f�Z�M�/�-�/�/�;�H�T�W�U�T�H�;�/�/�/�/�/�/�/�/�{�n�i�n�{ńŖŠŹ����������ŹŭŠŔŇ�{�ʼ��������żʼּ����߼ּʼʼʼʼʼ���ƳƹƷ������������$�0�5�3�,���������������������	���"�*�+�"��	����ììàßÝàçìîöõíìììììììì�����������������Ľнݽ���޽нĽ�����àÙÏÇ�z�o�f�c�l�n�z�~ÅØìúüóìà���������y�y���������Ŀݿ�������Ŀ����������k�a�Q�E�<�G�g�s����������������m�d�_�c�t¦³¿������¿²¦�0�-�$�"�$�&�0�6�=�?�=�=�0�0�0�0�0�0�0�0��x�r�n�r�y����������������������������������ʾ׾���������������׾ʾ��������������ͼ���!�.�5�6�0� ��ּ��������������~�v�w�y���������ſʿѿ�ݿѿ���������������������������������������������������������������
�������������¿µ²®²»¿����������¿¿¿¿¿¿¿¿����ܹϹ������ùϹܹ�����'�%�������������5�A�N�g�s���������s�g�A�5��W�G�C�F�O�c�x�������ûȻͻϻĻ������l�W���������������������������������������������������ʾ׾����׾ʾ��������������s�m�i�j�o�s�����������������������s�s�"������"�#�/�2�;�B�E�;�/�$�"�"�"�"�/�-�"������"�/�;�B�H�H�H�H�;�1�/�/��������������������� ������������������ŭŔ��|ŇōŔŠŲŹ����������������Źŭ�r�m�f�f�m�r���������������������r�r��}�w�u����������������������0�,������	�
��#�*�0�<�G�L�I�G�<�0�ѿϿĿ����������Ŀȿѿݿ����ݿԿѿѺr�o�n�o�r�}�~�����~�{�x�r�r�r�r�r�r�r�r�����������Ǻɺֺ�����������ֺɺ������������������	��"�)�/�2�1�/�"��	���𺽺��������ǺɺϺֺ����������ٺɺ��x�x�m�l�l�l�x�}���������x�x�x�x�x�x�x�x�׾ӾϾ׾۾������	������	�����׿	��������������	���$�!����	E�E�E�E�E�E�FFF$F1F9F8F:F5F%FFE�E�EṶ�����������ùϹܹ�ܹٹϹù�������������	� ���'�3�@�L�e�r�u�r�n�d�X�L�@�'��S�H�H�S�S�_�l�x�|�z�x�v�l�_�S�S�S�S�S�S���������������ͻл������	�����лû�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�āĊċĘĚĦĳĿ��������������ĿĦĚčāÓÇ�r�m�sÆÍÓàìù����ýùìçáàÓ�P�C�:�7�:�G�S�l�����������������y�l�`�P����ݽнƽɽн׽ݽ������%� �������������������'�(�2�.�(���������������	������������������������򽫽����������ĽƽϽĽ�������������������� �����ݽѽݽ������(�-�3�(�!��� ÓÒÊÇÁÇÓàìïñíìàÓÓÓÓÓÓ�������������ĿѿݿݿܿٿտѿǿĿ�������D�D�D�D�D�D�EEEEEED�D�D�D�D�D�D�D�E*EE(E*E7EAECEPERE\E_E]E\EPECE7E*E*E*E* J ` / n @ - : / 3 : D u + H 0 G B c J 6 = m + t ? [ 5 = 0  _ i ) $ ; 5 � * G A ; (  P 8 = z L E � M * V x B ; 9 Z M 0 8 8 C D ; 0 D V 9 ] 9 4 Q P  K  �  �  �  �  �  w  e  n  �  W  |  c  �  Y  -  w  �  �  l  (  �  �  �  �    �  B  �  S    �  �  �  Q  Y  1  i  �  �  .  �  �  �  �  D  �  �  �  �  B  �  �  ^  �  �  M  �  B  �  4  �  8  R  3  �    �  Z  D  �  9  �  �  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  +        �  �  �  �  �  �  �  �  �    p  _  N  =  +    :  4  /  '         �  �  �  �  b  4    �  �  d  M  C  9  �    C  f  z  ~  x  n  b  O  5    �  �  !  �  H  �  S    ^  E  -    �  �  �  �  �  �  �  n  Y  A  !    �  �  �  �    $  4  B  N  V  Z  [  X  O  <    �  �  �  `  (  �  �  }  �  �  �  �  �  w  c  J  ,  	  �  �  {  9  �      �  �  N  �  �  �  �  �  �  n  [  G  /    �  �  �  �  �  n  U  ;  "  4  8  =  A  3  /    �  �  m  b  |  �  �  �  l    �  �  �  �  �  n  Z  F  1      �  �  �  �  ^  4    �  �  e  .   �  O  D  7  (      �  �  �  �  �  �  m  Q  6      �  �    �  �  �  �  �  �  �  �  m  X  E  *    �  �  6  �  �  �  �  U  G  :  -          �  �  �  �  |  M     �   �   �      Z   6  J  j  �      $      �  �  �  Z    �  d  �  D  {  �  �  ~  �  �  �  e  :    �  �  �  �  m  ^  N  @  4  )        �  �  �  �  �  �  �  �  �  �  W     �  �  m  -  �  �  b    s  �  �  �  �  �  �  =      �  �  Z    �  p  �  3  ;  �  G  ?  8  0  #      �  �  �  �  �  �  �  u  \  A    �  �  �  �  �  �  �  �  �  �  �  t  Q     �  �  y  ?  �  ^     �      �  �  �  �  �  �  y  `  E  (  
  �  �  �  }  U    �  �  �  �  �  �  �  �  _  4      �  �  �  �  k  "  �  �   �  �  �  �  �  �  �  p  _  J  -    �  �  �  �  q  E  �  u    �  �  �  �  �  z  q  c  R  A  /      �  �  v  :   �   �   �  �  �                  �  �  �  `  &  �  �    �  ;  �  �  ^  8         �  �  �  �  ]    �  �  �  =  �  ;  *  >  m  |  �  u  X  .  �  �  �  B  �  �  U  �  w  �  P  �  ~  �  �  w  ?  �  �  L  �  �  7  �  �  V  �  �  >  �  c  �   �  �  �  �  �    4  A  H  W  V  A    �  �     �  f  �  I  �  i  w  �  �  �  �  q  ^  @  !  �  �  �  r  A    �  �  n  8  �    &  ,  /  .  &      �  �  �  �  �  �  �  ~  {  p  B  k  }  �  �  �  �  �  �  �  {  l  X  @     �  �  x    {   �  o  �  �  �  �  �  ]  +  �  �  p  /  �  ~  �  W  �  4  ]  8  �  g  S  \  S  =    �  �  �  =        �  �  �  �  a  (  V  C  1      �  �  �  �  �  �  t  d  T  B  -      �  �  E  ?      �  �  �  �    >  a  |  �  �  e  1  �  �  k  �  �  �  �  �  �  �  �  �  �  �  �  /  �  o  
  �  5  �  R  �  �  Q  �  �  �  �  �  t  Q  H  "  �  �  \  
  �  !  v  �  �  a  d  �  f  H  ,  �  �    6  �  �  
  �  �  3  �  !  k  �  W  �  �  �  �  �  �  �  u  N  %  �  �  k  �  `  �      n  �  �  �  �  �  �  �  �  s  _  K  7  "    �  �  �  �  �  �    
    �  �  �  �  �  �  �  �  �  �  �  t  q  r  s  t  u  n  b  V  H  9  +        �  �  �  �  �  �  �  �  �  w  X  <  :  8  2  ,  #      �  �  �  �  �  �  �  �  �  x  �  �  �  �  �  �  �  �  �  �  �  �  �  r  \  D  ,    �  �  �  �  }  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  "  �  �  �  �  �  �  �  �  �  r  Z  B  (    �  �  q  /  �  n  A  G  N  M  :  !  	  �  �  �  �  c  '  �  �  >  �  $  }  3  �  �  �  �  �  �  �  �  t  ^  H  2       �  �  �  _    �  �  �  �  �  �  �  �  �  �  �  p  I     �  �  �  e  #  �  n  m  b  W  L  @  0  !      �  �  �  �  �  s  U  /     �   �  =  8  3  .  )  $        	    �  �  �  �  �  �  �  q  Z  �  �  �  z  d  O  5    �  �  �  u  =  �  �  -  �  0  �  2  j  i  S  <  &    �  �  �  �  T    �  m    �  K  �  M   �  �  �  �  �  �  �  �  �  �  �  }  n  V  3  	  �  �  �  a      �  �  �  �  �  �  �  �  w  c  N  8  "    �  �  �  �  �  �  �  �  �  �  �  �  �  {  q  c  G  (    �  �  P  �  �  (  �  �  �  �  �  w  ]  @    �  �  �  �  o  H    �  �  �  Y  
  
�  
�  
k  
:  
  	�  	�  	�  	g  	"  �  E  �  	  Z  �  )  �  W  �  �  �  z  q  g  [  O  C  7  (      �  �  �  �  w  Q  *  �  �  �  �  �  �  �  l  <    �  v  ,  �  O  �  L  �  �  ,  @  @  A  >  2  '      �  �  �  �  �  �    `  @  '     �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  b  R  <  !  �  �  �  �  �  r  4    �  �  �  c  /  �  �  ]  �  O  �  &  �   �    *  ?  ,      �  �  �  �  Z  %  �  �  h    �  Y  7  #  �  �  �  �  �  ~  h  N  -    �  �  O    �  v  '  �  n  �        �  �  �  �  �  �  c  A    �  �  a  6  1  @  C  ;  l  �  �  �  z  _  ?    �  �  �  w  ;  �  �  U    �  �  �  �  �  �  �  �  �  �  v  V  .    �  �  d  (  �  �  b  "  �  	5  	5  	5  	5  	5  	5  	5  	5  	5  	5  	5  	5  	5  	5  	5  	5  	5  	5  	5  	5  y  q  i  a  Y  Q  I  A  9  1  )  !        �  �  �  �  �  &    	  �  �  �  �  �  �  �  �  w  [  ?     �  �  �  �  B      
    �  �  �  �  �  �  o  @    �  r    �  #  �  %  �  �  �  �  |  _  >    �  �  �  }  H  
  �  h    �  {  L  	9  	  �  �  X    �    /  �  �  I  �  N  �    w  �  (  z  u  h  D    
�  
�  
V  
  	�  	�  	x  	:  �  o  �  '  l  �  �  