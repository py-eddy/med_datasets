CDF       
      obs    ;   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?����n�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       Pߵ�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��j   max       =�x�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���
=q   max       @E��\)     	8   p   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����
=p    max       @veG�z�     	8  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @P@           x  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�d        max       @��`          �  3X   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��C�   max       >W
=      �  4D   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�G   max       B,��      �  50   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�{�   max       B,�9      �  6   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��   max       C�
�      �  7   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =��   max       C��      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          K      �  9�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /      �  :�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P4��      �  ;�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?����l�   max       ?�hr� Ĝ      �  <�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��j   max       >         �  =|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��G�{   max       @E��\)     	8  >h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vd�\)     	8  G�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @P@           x  P�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�d        max       @�,�          �  QP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C   max         C      �  R<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�*�0��   max       ?�f�A��     p  S(               D         R      .      0                  	      ,                     s      $                  6      ]               #         /            .      �         
      d         #N1��N���MݺZO� �P�/N{7�O�6Pߵ�N�O�O�=�O$v�O�̮O��NEa�N+��N�ճO�َN���O�PGP-��NP<NF�jO,YOqb�N?��P�O�E�O�CDN�a+N1`O��O���O.#0P+��N7/PtN�OT�N���M�N�ʍOA5�N���N�^�Oʗ�OO��N���Or��Oƕ�M���Pa�N�'2N���N���N֕jOx-N�8HN͜O �K��j���
�D���ě�;�o;ě�<o<D��<T��<T��<e`B<u<�C�<�t�<�t�<���<�1<�1<�9X<�9X<�j<�j<�j<���<�/<�/<�`B<�`B<�h<�h<�=o=+=+=\)=\)=t�=t�=�P=�w=<j=e`B=ix�=ix�=y�#=y�#=�o=�7L=�7L=�O�=�t�=���=���=�^5=ě�=���=���=�;d=�x�ZV[dgptytog[ZZZZZZZZ^[\hmt����������th^^��������������������#0<IUbn|xobZU<0#EEGdt����������th[PEfelmuz|}�zvnmffffff����������������������)BRZXdt|rg[>*��������	pmnoq�������������zp�������
�����~~~����������������� #7BKHB5)	�����������������������������������������LOZ[htyytph[OOLLLLLL!)6BOhzztjhlhc[LB6)!1269=BOS[hihf[OGB611�� 
/9<EHJI><%#
�#%#/5<H\ilh^UH</�������������������;779<?CHTJH<;;;;;;;;5126<FHJU^adbaUSH<55z������������������z����	


	����������������*.*
�������xsst�������������xtxfdf_hsw���������kf����������������������������������������khgjmnvz|�������zmkk����
#*0-
�����
#,/-)# 
��4/1415Nt�������tgN<4=ABCNVVSNCBB========����5BG?>FXfe[M!��MLNS[fgt�����tgd[TNMqtz�������������yutq�������������������� ��     ��	"/;;?=;5/'"	�#%%#
 �����
#,-,/:<HJTHFE</,,,,,,{wy}���������������{.++1<HUalnsnmfaUH</.��������������������_ZWXa^amz�������zma_�������	�����������������������������������������������!"��������������������������
 �������
���
!#&'%#
	��������������� 

���������������������������������������������ZUVYadntz~��|zuniaZ�����������������������������������������@�L�Y�_�e�o�o�j�f�e�e�Y�L�I�@�?�=�8�@�@��������������������������������������������������������������v�f�G�Q�J�P�Y�f����������¾˾оξʾ�������f�S�G�G�M�Z����������������������������������������˾4�A�M�Q�Z�`�f�k�f�M�(��������(�4����0�Q�R�B�$���Ƴƌ�h�6����O�rƳ��������������ùòñøù�������������������tāčĚĦĳĴĴıĩĚčā�t�g�d�d�d�h�t������*�,�5�/�*�%����������������(�5�N�g�n�s�z�s�p�g�d�Z�N�5�(�#����"�;�G�T�m�u�x�m�`�T�J�;�.�)�#�����"ŭŹ������������ŹŲŭťŭŭŭŭŭŭŭŭùù��úùìàØàãì÷ùùùùùùùù�:�<�F�I�R�J�F�:�3�-�)�+�-�-�:�:�:�:�:�:����������������������r�Q�Y�]�r�y�v�x��ÓàæìùýûùóíìçàÝÙÓÎÈÓÓ����&�)�,�5�:�5�1�(����������s������������������������s�g�X�N�R�d�s�Z�s�����������n�g�Z�N�5������#�2�ZE*E7ECEPESEPECE9E7E*E*E)E*E*E*E*E*E*E*E*�F�S�_�g�d�_�S�F�:�:�:�E�F�F�F�F�F�F�F�F������������
������������������������������������������s�o�i�q�s�����������A�M�Z�_�f�f�f�Z�M�A�A�>�A�A�A�A�A�A�A�AÇàù�������������ïàÇ�z�W�?�I�UÇ�ùϹܹ߹����	������ܹϹ͹��������������/�<�H�U�g�n�n�a�X�N�H�<�/�#�����#�/����'�,�)�'���������������ÓààêêàÚÓÌÎÓÓÓÓÓÓÓÓÓÓ���������	��� ��������������������پ����ʾ׾��	�"�,�&�����ʾ����������`�m�y�|���}�y�w�m�`�T�G�<�;�3�;�G�T�]�`����/�;�H�T�V�F�.�"��	�����������������A�N�P�N�J�A�5�(�!�(�5�7�A�A�A�A�A�A�A�A�
�#�I�W�`�d�t�v�n�<�����ĿĚăĚ�����
�[�h�n�t��t�s�h�_�[�O�B�:�9�@�B�I�O�Y�[������������������������������������������������������
����
�������������������������T�a�m�x�z�����z�y�m�a�]�T�N�H�D�?�B�H�T���������������ľʾ׾��������׾ʾ�¦²·¿¿¿¿²§¦�(�4�A�M�Z�f�l�i�Z�4�(������
���(��������$�)�$�������������������M�Z�f�s�s�s�s�m�f�Z�M�I�A�;�A�D�M�M�M�MŔŠŭŹ��������������ŹŭŠŔŒŋōŒŔ�������������������������z�q�o�u����������������������������������������������������ּ���:�G�L�K���ʼ������q�i�o���t¥¥�}�t�o�l�t�t�t�t�t�tEuE�E�E�E�E�E�E�E�E�E�E�E�EwEuEmEuEuEuEu�I�I�I�L�U�Y�b�h�n�{�|��{�y�p�n�b�U�I�IǡǭǭǶǶǭǪǡǔǈǃ�~ǆǈǔǜǡǡǡǡD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D}DuDuD{D��_�l�x��������������x�t�l�_�[�X�_�_�_�_������������	��	���������������������׻��ûлܻ������ܻлû��������������� ; Q ` H 6 o i f ^  *  C = d P R ^ m W : d E   0 ; V 9 M + H E n . ? T n U = X C A ] E  6 % + F � X [ B O + 1 F o <    L  L    �  �  �  �  �  �  6  d  �  �  d  u  �  R    �  �    X  V     �  d  R  +  9  �  ?  B  ~  q  N  V  �  h  9  $  �  �  >  �  �  �  �  �  �      �  �  �  �  �  �  9  s��C�:�o�t�<��
=�t�<49X<�1=��<�h=m�h<�h=}�<�<�1<�h<���=,1<�=#�
=��=L��<�`B<�/=@�=T��=o>n�=m�h=��=C�=\)=<j=Y�=q��=�9X=��>�=D��=D��=,1=y�#=�^5=�O�=��=�/=�E�=��-=�v�=�l�=�hs>W
==�-=�E�=��=��m>O�;=�h=�F>�uB	=�Bi�B��B&�`B{�A�V?B"�B�"B�:B ��B�B��BG�B�B!�$B�B�uB|�B��B�tB�tB�B0�B'�B�1BnMB9�B�=BL�B"!�B ��A��{B�aB B	t�B�B�B	a`B
�LB!�B�mA�GB$�EB�B�rBe�B�}A��B��B*%'B=~B��B�B�8B�B��B,��B�zB�B	>QB�]B�LB&F.B��A��eB!��B�FB9"B2@B��B��B5B�B"=VB��B��B��B�cB�]B��B��B0dB;TB��B})B?�B@BB3B!�%B �kA���B�bB@B	SQBK�BĂB	[�B
��B �/B��A�{�B$?�B�B��B?#B�6A��B��B)��B�-B�uB�B�HB>SB�B,�9BQB�rA�W�?��A���@�BAF+�A���A9�BʹAΡ�A�x�A��RA���AcņA��VĄ�@}@���A˞�A�}A��5A�'YC���@�^�A�7�A���A=ˈA�B�>��A��d?~�KA�*B�!AT��Ah�7A�ʤA��KA��A�j`A��@�3kA��jA���AQ6�A���A8�A���A?�A���A�F�A!��A*PA�GC�
�A�%B��C�Ւ@�$:A�=@���A�F?���A�~k@��AF�A�&+A:͛B=�A΂�Aބ�A��BA��2AcA���A�p�@{�G@�A�ÂiA��tA���A��gC���@�XA�{,A���A=dA·�=��A��?�A�qjB��AT��AiGA��{A��-A��A��JA��@�
A�{aA���AQ�A���A8�TA��VA>��A���A�|A"��A��A�QC��A�{�Bq�C�ס@��cA�b@��               E         S      .      0                  
      ,                     t      %                  6      ^               $         0            /      �               d         $            )   )         K               !            !         +   +                  7   !               #      +      ?                                    !      7                                                               !            !            )                                          !                                          !      /                        N1��NJKvMݺZOz�~O-KYN{7�N�G�O��kN6G�O& zNS�:O*O��NEa�N+��N�ճO�َN��7O�O_j�P+�NP<NF�jNx�N���N?��O���O���O��N�a+N1`N��OKkOrO��<N7/O�ژOT�N���M�N�4�O�N��qN�^�O��WO$kN���O_�Oƕ�M���P4��N�'2NX%vN���N֕jO$9N�8HN�ʠO%Z  �  �  �  ;  �  �  �  t  �  5  1  �  �  a    �  �  7  �  �  }  �  �  2  M  �  
�  e  �  <  "    �    �  �  	�    M  �  �  r  �  �  �  a  E  6  t  r  �  �  u  �  	�  �  �  H  	м�j�D���D��:�o=,1;ě�<t�=y�#<�o<�/<�1=C�<�C�<�t�<�t�<���<�1<�j<�9X=#�
<���<�j<�j=+=\)<�/=�-<�h=��<�h<�=+=t�=t�=D��=\)=��P=t�=�P=�w=@�=y�#=m�h=ix�=�t�=�+=�o=�C�=�7L=�O�=�Q�=���=��w=�^5=ě�>   =���=�G�=�ZV[dgptytog[ZZZZZZZZhjtt�����thhhhhhhhhh��������������������# #02<GIYnjb_UI<0%$#f`__gst����������thffelmuz|}�zvnmffffff��������������������)5>BHLNNMB5������������~|}���������������~~������������������������������������� #7BKHB5)	�����������������������������������������LOZ[htyytph[OOLLLLLL!)6BOhzztjhlhc[LB6)!356;?BNO[c][OCB63333�� 
/9<EHJI><%#
�"%*/<HLRUZ^]WUH</&#"��������������������;779<?CHTJH<;;;;;;;;67;<HPUZUSH<66666666������������������������	


	�������������������������tst�������������yuxtrntv������������xtr����������������������������������������mihkmqz�������zmmmmm��
#')/*#
�����
#)-+(#
���?:77:BN[t������tg[L?=ABCNVVSNCBB========�����)+1462+)��MLNS[fgt�����tgd[TNMqtz�������������yutq����������������������	"/6;=;2/#"	����
"#$#
��,-,/:<HJTHFE</,,,,,,������������������../4<AHUagmhaULH<3/.��������������������`[XYabmz��������zma`�������	�����������������������������������������������!"��������������������������
 �������
���
!#&'%#
	�������������������������������������������������������������ZUVYaentz}��{zsnjaZ�����������������������������������������L�Y�e�f�g�e�Y�L�G�E�L�L�L�L�L�L�L�L�L�L�����������������������������������������r�����������������������i�\�[�T�Y�f�r�f�s���������������������s�r�f�e�a�d�f���������������������������������������˾4�A�M�N�Z�]�f�Z�M�A�(���	����(�1�4ƧƳ����������������ƳƧƚƒƁ�w�|ƎƚƧ��������������ù÷ôù������������������āčĚĦĩĭīĦĠĚčā�t�s�n�q�t�uāā����'� ����������������A�N�Z�`�d�_�Z�V�N�A�5�3�(�'�!�$�(�5�7�A�"�;�G�T�m�u�x�m�`�T�J�;�.�)�#�����"ŭŹ������������ŹŲŭťŭŭŭŭŭŭŭŭùù��úùìàØàãì÷ùùùùùùùù�:�<�F�I�R�J�F�:�3�-�)�+�-�-�:�:�:�:�:�:����������������������r�Q�Y�]�r�y�v�x��ÓàáìùüùùñìàßÜÓÒÍÓÓÓÓ����&�)�,�5�:�5�1�(����������������������������������s�p�e�k�s�������N�Z�s�������������|�g�Z�W�(����%�5�NE*E7ECEPESEPECE9E7E*E*E)E*E*E*E*E*E*E*E*�F�S�_�g�d�_�S�F�:�:�:�E�F�F�F�F�F�F�F�F�������	����������������������������������������������������t�s�o�s�����������A�M�Z�_�f�f�f�Z�M�A�A�>�A�A�A�A�A�A�A�Aàìù��������������������ùçàØÑÕà�Ϲܹ޹��������ܹϹ��������������ù��/�<�F�H�U�U�Q�H�<�/�#�������#�*�/����'�,�)�'���������������ÓààêêàÚÓÌÎÓÓÓÓÓÓÓÓÓÓ�������������	���������������������پ��ʾ׾��	���������ʾþ����������m�x�y�~�|�y�t�m�`�T�G�?�;�8�;�G�T�`�c�m���	��"�/�:�G�E�>�9�)�"��	�������������A�N�P�N�J�A�5�(�!�(�5�7�A�A�A�A�A�A�A�A�
��#�5�<�G�N�Q�K�I�0�
���������������
�[�h�n�t��t�s�h�_�[�O�B�:�9�@�B�I�O�Y�[������������������������������������������������������
���
�	�������������������������H�T�a�i�m�u�z�~���z�r�m�a�T�Q�H�G�B�F�H�ʾ׾�������׾ʾ��������������Ǿ�¦²·¿¿¿¿²§¦�4�A�M�V�_�_�Z�M�A�4�(����
�
���(�4��������!�&� ��������������������M�Z�f�s�s�s�s�m�f�Z�M�I�A�;�A�D�M�M�M�MŔŠŭŹ��������������ŹŭŠŔœŋŎœŔ�������������������������z�q�o�u����������������������������������������������������ּ��.�>�?�!����ʼ������y�s�{���t¥¥�}�t�o�l�t�t�t�t�t�tE�E�E�E�E�E�E�E�E�EyE�E�E�E�E�E�E�E�E�E��I�I�I�L�U�Y�b�h�n�{�|��{�y�p�n�b�U�I�IǡǭǭǶǶǭǪǡǔǈǃ�~ǆǈǔǜǡǡǡǡD�D�D�D�D�D�D�D�D�D�D�D�D�D{DzD{D}D�D�D��_�l�x��������������x�t�l�_�[�X�_�_�_�_��������������������������������������׻��ûлܻ�����ܻлû��������������� ; F ` I ( o \ : c  *  C = d P R Z m ) 4 d E ' ) ; 8 6 ; + H 7 d 3 @ T / U = X > B ] E  < %  F � R [ / O + ! F W :    L  t      r  �  I  *    `  c  f  �  d  u  �  R  �  �  �  �  X  V  }    d  )    b  �  ?    �  F  '  V  �  h  9  $  �  W    �    j  �  �  �      �  U  �  �  c  �  �  g  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  �  �  �  �  �  �  �  �  �  �  �  u  b  K  .    �  �  �  �  x  �  �  9  r  �  �  �  �  �  �  �  �  �  c  /  �  �  �  f  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �       5  ;  6  %    �  �  �  �  �  �  ~  @    �  K  *  s  �  �  %  V  �  �  �  �  �  �  �  �  i     J  ]  I  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  U  8    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ^  (  �  �  )  U  U  C    �    y  �  �  �  D  q  d  (  �  O  �  }  B  L  �  �  �  �  �  �  �  �  �  �  {  V  $  �  �  <  �  W  �  g  B  �  �  �    /  5  4  1  '    �  �  @  �  E  �    S  �  �  �  �          &  -  1  .      �  �    1  �  ^   �  �    I  r  �  �  �  �  �  �  �  o  *  �  n     �  �    �  �  }  n  \  H  +    �  �  �  w  `  K  5      '    �  �  a  [  V  Q  K  F  A  ;  4  .  '  !      �  �  �  a  7          �  �  �  �  �  j  L  /  ,  8  "  	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  b  I  4       �  �    '  M  �  �  i  M  J  U  <  "    C  =  4  =  d  i  e  f  �  �  A       +  1  6  6  4  )      �  �  �  |  R  )    �  o    �  �  �  �  �  e  :    �  �  �  �  �  f  >    �  �  M  �  �  �    J  |  �  �  �  �  �  �  �  p  7  �  p  �  F     �  o  z  {  q  h  V  :    �  �  �  �  �  �  �  �  �  v  "  �  �  �  �  �  �  �  �  �  �  x  `  I  /    �  �  �  �  �  i  �  �  �  �  �  �  �  w  k  _  S  F  9  ,    !  '  -  3  9  l  �  �  �    "  1  2  .  #    �  �  �  S  �  f  �  +  �  �  �  �  �    2  L  C  2    �  �  �  y  J    �  �  �  :  �  �  �  �  �  �  �  �  �  �  �  �  �  �    w  [  :    �  a  �  	4  	n  	�  	�  	�  
	  
N  
r  
  
n  
I  
  	�  	  B  /  U  �  [  B  &  5    �  �  �  �  r  {  G  E    �  i  �  x  �   �  q  n  j  x  �  �  �  �  �  X  (  �  �  t    �    �  �    <  ,      �  �  �  �  �  �  �  �  �  �  �  x  G     �   �  "  #  $  %  $    
  �  �  �  �  �  �  {  ]  >    �  �  �  �              �  �  �  �  �  a  8    �  n    �  H  �  �  �  �  �  y  _  <    �  �  �  �  n  /  �  �  i  W  v              �  �  �  �  a  4    �  �  &  �  �    (  f  x  �  �  �  �  �  �  �  |  u  h  N    �  D  �  *  D  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  	  	9  	2  	  �  	r  	�  	�  	�  	�  	e  	,  �  |  �  �  �    u      �  �  �  �  �  h  M  2    �  �  �  �  �  �  �  v  d  M  F  >  5  ,  #      	  �  �  �  �  �  m  K  *  	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  v  n  `  {    v  e  R  <  (    �  �  �  �  �  d  =    �  �  �  Z  f  n  r  k  ]  C    �  �  �  E    �  Y  �  i  �    w  $  j  �  q  7  �  �  u  7  �  �  �  z  T  :    �  �  �    �  t  8  �  �  ]    �  �  E    �  �  �  2  �  �  ,  �  A  �  �  �  �  �  �  �  �  �  g  *  �  �  I  �  �  +  �  �  �  =  L  X  _  ^  T  D  +    �  �  �  :  �      �    :   �  E  4  !    �  �  �  �  �  q  a  T  E  2  #      �  �  �    /      �  �  �  �  �  r  F    �  �  E  �    [  �   �  t  c  O  /    �  �  �  �  �  h  $  �  |    �    @  �  �  r  c  T  E  6  '    	   �   �   �   �   �   �   �   �   �   �   �   v  +  �  �  �  �  Z  ;    �  �  ,  �  Z  
�  
�  
   	G    o  �  �  �  i  Q  9  !  	  �  �  �  �  �  }  J    �  �  1  �  �  -  ;  U  s  Y  <    �  �  �  e  1  �  �  �  N    �  {    �  �  �  g  A    �  �  �  i  <    �  �  �  {  P  !  �  �  	�  	�  	�  	W  	  �  �  �  /  �  �  C  �  �  2  �  L  �      n  �  �  �  �  �  �  �  R    �  +  �  �  �  5  
B  �  =  g  �  �  {  l  `  S  G  7  %    �  �  �  `  2    �  �  �  t  2  >  E  '    �  �  �  �  t  \  D  '    �  �  j     �  d  	�  	�  	�  	t  	E  	  �  �  }  4  �  l  �  �    �    |  �  M