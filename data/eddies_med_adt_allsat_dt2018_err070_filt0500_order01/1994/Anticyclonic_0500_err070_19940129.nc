CDF       
      obs    2   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��1&�y      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       MҾK   max       PC��      �  t   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��C�   max       =���      �  <   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @FxQ��     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vw�
=p�     �  '�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @P@           d  /�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�L`          �  0   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��`B   max       >2-      �  0�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�V   max       B2�      �  1�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~L   max       B2�d      �  2`   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?7f   max       C��      �  3(   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�u   max       C�h      �  3�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          p      �  4�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          1      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  6H   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       MҾK   max       P)�?      �  7   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Fs����   max       ?��l�C��      �  7�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��o   max       =�l�      �  8�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @FxQ��     �  9h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @vw�
=p�     �  A8   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P@           d  I   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���          �  Il   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�      �  J4   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Ov_خ   max       ?����     �  J�      5                        ]            	                  (                     (         	   &                           	                        p      JNT��P'KN7��N�%O��N.A�O�OgOW�GOɉ�O�?�N���O.f/N��[N��#N%}OR�N�;�Oq_O���PC��O�`)N�zO�-�O��jO '�P!gjO�NNEwuN��O���N��QO�� O�ƛOK\EO$X#Os]
Nr-�OqR�NlU�N)�|N$h0N���N�|�MҾKM��[NJثO�3MN*@P ����C�%   :�o:�o;�`B;�`B<o<o<t�<49X<D��<T��<T��<e`B<e`B<�o<�t�<�t�<���<���<�1<�9X<�9X<���<���<�/<�`B<�<��<��=+=C�=t�=t�=t�=�P=�P=��=8Q�=8Q�=@�=D��=L��=P�`=]/=ix�=u=�%=�C�=���hhqt�����thhhhhhhhhh48B[ht��������hOD=74ktt�������tkkkkkkkk��������������������p�}���������������tp:<HUU_UHA<::::::::::�}������������������|~������������������������������������}}����������������������
'/5A@</#
��#/6<>?<:1/&#)-6O[hilmih_[OMB?76)jmz�����~znnmmjjjjjj��������������������xwz�����|zxxxxxxxxxx���)356;:4����������������������5,**,5BNY[`db\[NFB55" /<HUgjf[TOVUH<5"�����!5?<)����������������������!#03<DFC<;0)'#����������������������������������������������������������������)16;NUNB5������5BHHJHEC)��2/6BOZYOB62222222222��������������������	)5BOPOMIB5)	������������		"%/5:<872/("	YX[gt��������tgefc[Y 0<IPVURIC=0#����
#,-/0/-#
��-1101<HUZahkhaUH<2/-������������������������!##���()268;BOSONIB;6)((((������� ��������������������������������������������������������������������������������������������?@BO[`[OOB??????????����uh`dhu�������������������
������{nkns{���������������������������������L�Y�e�j�l�e�Y�L�K�G�L�L�L�L�L�L�L�L�L�L���ʼҼټؼҼ̼ü����������r�[�Z�f�������y�����������y�v�y�|�y�q�y�y�y�y�y�y�y�y�G�T�U�\�^�^�T�G�D�;�<�B�G�G�G�G�G�G�G�G�"�;�H�P�S�\�]�T�H�2��	���������� ��"�m�t�y�n�m�a�X�Z�a�i�m�m�m�m�m�m�m�m�m�mE�E�E�E�E�FFFFFE�E�E�E�E�E�E�E�E�E��/�<�H�L�U�P�H�A�<�0�/�#�"������#�/���� ���'�*�0�,�'��������ܻ����B�O�[�h�t�z�z�x�[�B�7�)�'�����'�6�B�G�T�`�m������������y�`�T�K�<�/�%�'�1�G����������������v�s�f�d�f�i�s����������׾������׾ʾ����������������ʾ̾�čĒĘčĆā�~�t�h�]�h�tāĂčččččč�ĿѿҿӿѿοĿ��������������ĿĿĿĿĿ��A�N�X�S�N�A�5�)�5�8�A�A�A�A�A�A�A�A�A�A�\�h�u�}ƁƎƚƟƦƚƓƎƁ�u�h�a�[�P�T�\����������� ������������������������������������"�$������������������������������������������ùìôõ���޿��ѿ���ݿѿĿ����y�`�G�;�5�;�L�m�����Ϲ�����+�3�-�'������ܹù��������ϼ����������������r�g�f�^�f�r�y�����tāĚĦįĵĿ��Ŀīčā�t�h�O�N�[�^�m�t�Ľݽ����������ݽĽ���������������ùý��������þùìåàÓÎÈÐÙàìóù�#�<�P�błōŎ�{�b�U�<�#�����
� ��#�Z�f�s�s�v�w�s�h�Z�M�A�4�(����(�4�A�Z�����������������~������������������������������
����������ܹعܹ������)�5�B�[�f�z�t�g�N�B�5�%��������	��"�.�;�G�P�G�F�;�.�"��	����������"�/�;�H�W�W�T�F�;�/�"��	�������������"�����������������s�g�Z�V�Y�[�_�g���������ּ�������������������Ѽɼ̼���(�.�5�<�A�D�L�A�5�(����������Z�g�s���������������������s�i�c�`�Z�U�Z�A�M�P�S�R�M�J�A�:�4�(�$�(�0�4�;�A�A�A�A�x���������ûȻû��������x�l�_�C�=�S�[�x�ллܻ������������ܻӻл̻ͻллло��������������������������������������������)�0�)��������������������
�
�
�������������������������������������������������������y�r�x�y�����#�/�1�<�=�<�/�#�"��#�#�#�#�#�#�#�#�#�#�������úɺ������������������������������	������������	����	�	�	�	�	�	�	�	�	D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DyDxD~D�D��������� ����������������Ｄ���ֽ�!�:�B�>�.�����ʼ����������� H < � ` F V W 3 2 2   - 6 Y M Z Y M B ; o E 0 > ` F ; 9 f > O u Q k X # ? \ z u B g ` q P o r ? \ _  x  �  �  �  �  _  z  :  �  �  �  �  �  �  �  .  �  �    j  �  �  �  |  �  =  �  �  �    �  (  >  �  �  Y     �  A  �  J  o  �  Y    G  �  �  e  ��`B=Y�;�o;�o=�P<t�<�/<�<���=��='�<�t�<���<�9X<��
<�9X<���=+=o=q��=T��=P�`<�=<j=P�`=L��=�C�=P�`=C�=�w=�hs='�=�o=H�9=e`B=]/=ix�=,1=�%=Y�=]/=L��=ix�=ix�=��=q��=�%>2-=���>��Bj B�eB
6~B�7B�YBQpB=jB��B!�B��B;�B&B�A�E�B �"B�,B�B½B�B�BB#B��B%ʟBɜB!,]B!�LB�BT�BNqB!3B�BLNA�VB	̌B%Z BK�B^BB94B�BkB'hByPB
,B,-�B$B��B2�B�]B(ʥBFuBQyB�,B	��B��B��Bs6B@0BB!��B��B�BAB
�A�sB ٰB�zBHkB�B¢B�BTB��B&AB�sB!@B">B��BEBr(B ݔB��BAA�~LB
.�B%B�BN.B��B3�B�]B?\B5�B�[B��B,D�B>�B�[B2�dB�XB)B?�?�X@�Amx�Ae�?A�p�A��C��A��J@���Aع�Aj4�AESARJ�A�$�Ax]zA���B��A�i.B�A�f Aq
U?7f@�#A�1A+Q~A�&A�dA<u�A��?3.DA��A_:�A���A�(�A�A�ASA��_A9�@��z@�HHAIöA��A�]gA*�A�L@)�AY��C��L@Y�FAW�?��@�NAm	uAe�7A���A��C�hA@��A�{Aj�AEOAQ�YA�uAx��A�v�B�AЃpB!AA�~�Al
?�u@�A���A,�RA�P3A�5A>�$A�b�?OB�A�HSAa�A��sA��vA0�A���A�~�A:��@���@��AI	A��GA��A A�@0�}AZ��C���@\�A�5      6                        ]            	                  )          	            )         	   '                           	                        p      J      )         '               !   #                           !   1   #      !         +   !                  !               !                                 +                                                               1         !         +   !                  !                                                N4��ORKN7��N�%Oq�N.A�O�N�V�O7�O�O;0N��;O�kNQ�NSt0N%}OR�NSh/Oq_N�qP)�?O3[N�zO�-�O9K�N݊eP!gjO�NNEwuN��O��!N��QOF�"O�ƛN��bO$X#OK��N��OY��N<��N)�|N$h0N���N|�MҾKM��[NJثO�N�PO7:�  �  �  �  �  �  �  <  �  �  f  |  ?  F  �  �  �  8  "  �  *  �  `  �  �  �  �  �  v  �  �      �  �  �  O  �  �  7  �  �  F  U  4  �  T      L  	Ӽ�o<���:�o:�o<�9X;�`B<o<�o<49X=aG�<�9X<e`B<u<u<u<�o<�t�<�9X<���=�P<ě�=+<�9X<���=C�<�h<�`B<�<��<��=\)=C�=#�
=t�=49X=�P=�w=�w=<j=<j=@�=D��=L��=T��=]/=ix�=u=�`B=�O�=�l�kirt�����tkkkkkkkkkkLJKO_hr�����|tlh[YPLktt�������tkkkkkkkk����������������������������������������:<HUU_UHA<::::::::::�}�����������������������������������������������������������������������������������
#,/262(#
�#$/4<=><80/-#456BO[hhkkhh[[OCB;64lmz�����|zrpomllllll��������������������xwz�����|zxxxxxxxxxx���)356;:4����������������������5,**,5BNY[`db\[NFB55+*+/:<=HNUUURHC<6/++�����1:6)�������������������������!#03<DFC<;0)'#����������������������������������������������������������������)16;NUNB5������5BHHJHEC)��2/6BOZYOB62222222222��������������������	)5BMONLHB5)������������")/28:64/"YX[gt��������tgefc[Y #0<860*#����
#,-/0/-#
��94238<@HUaehifa_UH<9����������������������� 	""���.469<BOROLIB86......������� ��������������������������������������������������������������������������������������������?@BO[`[OOB??????????����uh`dhu������������������


	������lnt{������{nllllllll���������������������L�Y�e�i�j�e�Y�L�L�H�L�L�L�L�L�L�L�L�L�L�����������¼��������������|�r�o�p�r����y�����������y�v�y�|�y�q�y�y�y�y�y�y�y�y�G�T�U�\�^�^�T�G�D�;�<�B�G�G�G�G�G�G�G�G�"�/�3�:�0�/�(�"���	������������	��"�m�t�y�n�m�a�X�Z�a�i�m�m�m�m�m�m�m�m�m�mE�E�E�E�E�FFFFFE�E�E�E�E�E�E�E�E�E��/�<�H�I�H�G�<�<�/�#� ��#�%�/�/�/�/�/�/���������(�,�'�$��������߻����B�O�V�[�c�d�[�X�O�B�6�+�)�'�)�+�6�=�B�B�`�m�y�����������y�t�m�`�T�G�B�C�G�L�T�`�s�����������������y�s�f�f�f�n�s�s�s�s�׾�������׾ʾ��������������ʾվ�čďĖčĂāĀ�t�h�`�h�tāĉčččččč�ĿʿѿҿѿͿĿ��������ÿĿĿĿĿĿĿĿ��A�N�X�S�N�A�5�)�5�8�A�A�A�A�A�A�A�A�A�A�\�h�u�}ƁƎƚƟƦƚƓƎƁ�u�h�a�[�P�T�\����������������������������������������������������"�$���������������������������	����
�����������������������ѿݿ���ݿĿ����y�`�G�;�9�>�O�m�����ѹܹ������������ܹϹù������ùϹܼ����������������r�g�f�^�f�r�y�����tāĚĦįĵĿ��Ŀīčā�t�h�O�N�[�^�m�t����������ݽнĽ����������Ľнݽ��àìù��������ýùìãàÓÐÊÒÓÜàà�#�<�P�błōŎ�{�b�U�<�#�����
� ��#�Z�f�s�s�v�w�s�h�Z�M�A�4�(����(�4�A�Z�����������������~������������������������������
����������ܹعܹ������)�5�B�[�d�w�t�g�N�B�5�'��������	��"�.�;�G�P�G�F�;�.�"��	������������	��"�/�;�H�P�Q�H�B�;�/�"��	�������������������������s�g�Z�V�Y�[�_�g���������ּ������������ڼּҼּּּּּּ���(�.�5�<�A�D�L�A�5�(����������g�s�������������������������v�s�k�f�e�g�A�M�N�M�L�A�7�4�(�'�(�3�4�>�A�A�A�A�A�A�x�����������ûŻû������x�l�_�E�?�S�_�x�лܻ޻�����������ܻԻлλллллло��������������������������������������������)�0�)��������������������
�
�
���������������������������������������������}�y�t�y�z�������������#�/�1�<�=�<�/�#�"��#�#�#�#�#�#�#�#�#�#�������úɺ������������������������������	������������	����	�	�	�	�	�	�	�	�	D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D������������������������������������ʼּ������������ּʼļ��������ļ� B % � ` M V W : +  , * . ^ G Z Y = B  p 6 0 > N C ; 9 f > M u L k < # " v v r B g ` f P o r ( o ;  W  �  �  �  V  _  z  �  �    �  �  D  ~  r  .  �  o    �  �  �  �  |  �  �  �  �  �    �  (  �  �  �  Y  �  a  #  �  J  o  �  �    G  �  -  _  �  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  �  �  �  �  �  �  �  �  �  �  h  B    �  �  �  =  �  �    p  �  �  �  �  �  �  �  �  �  �  �  �  N  �  |  �  y  �  f  �  �  �  �  �  |  r  r  x  ~  �  �  �  �  �  }  s  i  `  V  �  �  �  �  �  }  {  w  s  n  j  f  b  ]  W  Q  K  E  ?  9  �  �  �  !  M  v  �  �  �  �  �  �  �  p  G    �  s  *  A  �  �  �  �  �  �  }  y  u  q  m  i  e  a  \  X  T  P  L  H  <     �  �  �  �  t  (  #        �  �  �  �  W    �  H  �    B  ^  v  �  �  �  �  �  �  f  D  !  �  �  �    �  4  �  �  �  �  �  �  }  p  ]  D    �  �  �  �  �  �  k  3  �  	�  
  
�  
�  c  �  �  4  T  e  V     �  b  
�  
.  	`  :  X  �        _  u  x  {  {  y  o  ^  G  (  �  �  �  <  �  |  :  '  -  3  :  >  9  5  0  (      �  �  �  �  �  �  y  c  L  2  :  @  E  C  >  7  +    	  �  �  �  �  ^  0     �  �  t  �  �  �  �  �  �  �  �  �  �  �  �  �  z  `  =    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  i  �  �  �  �  �  �  �  �  �  �  �  �  r  c  T  F  7  *      8  2  ,  %        �  �  �  �  �  �  �  �  �  �  c  7    �  �  �  �  �              �  �  �  �  P    �  �  x  �  �  �  �  �  �  �  �  �  j  O  4    �  �  �  �  Y     �  �  )  l  �  �  �      )  '    �  �  �  <  �      �  `    �  �  �  |  l  Z  T  g  y  s  [  1  �  �  G  �  �      ?  >  8  9  ?  Q  Z  `  [  O  ?  &    �  �  w  ?    �  �  �  �  �  �  �  �  v  m  d  [  S  N  H  C  >  2  #        �  �  �  �  �  �  �  �  �  �  �  �  l  F  "    	        �  �  �  �  �  �  �  �  �  �  �  �  d  B    �  w  �  Y   �  �  �  �  �  �  �  �  �  \  &  �  �  �  K  '  �  �  �  h  �  �  �  �  �  �  �  �  �  r  o  �  �  �  `  6    �  d  �    v  k  [  8    
  �  �  �  �  *  3  3  '  �  �  D  �     l  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  p  g  ]  T  K  �  �  �  �  �  �  �  �  �  k  ^  Y  S  J  A  7  ,  #      �       �  �  �  �  �  �  y  V  '  �  �  _    �  U  �          �  �  �  �  �  �  �  �  �  �  n  T  :        �   �  �  �  �  �  �  �  �  �  �  �  �  x  R    �  P  �  K  �  %  �  �  �  �  x  X  4    �  �  �  z  Z  :  (      �  �  y  s  s  {  �  �  �  �  �  �  �  �  �  w  ^  A    �  �  b  1  O  ?  *    �  �  �  �  �  �  �  t  e  Q  -  �  �  5  �    �  �  �  �  �  �  �  �  �  x  U  -  �  �  �  q  J  2  0  E  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  3  7  0      �  �  �  �  �  �  �  �  t  V    �  i    �  �  �  �  �  �  �  �  �  �  �  j  P  7  '        �  �  �  �  �  �  �  �  �  s  a  O  8  !  	  �  �  �  �  �  s  ^  I  F  C  ?  ;  7  4  0  ,  )  %  #  !                   U  G  :  /  )  #  +  ?  R  `  n  }  �  �  �  �  �    *  G  
    %  2  1  .  +  $           �  �  �  �  �  1   �   q  �  �  �  �  �  W  ,  �  �  �  p  =  	  �  �  d  *   �   �   t  T  P  L  H  D  @  <  8  4  1  /  /  /  /  /  /  0  0  0  0      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  "  I  x  �  �  �      �  �       �  �  �  �  
$  �  G  I  J  H  =  2  #    �  �  �  �  �  �  k  G  $  �  �  �  	M  	F  	4  	%  	L  	�  	�  	�  	�  	�  	�  	�  	M  �  b  �  :  5  �  �