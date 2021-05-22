CDF       
      obs    4   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?ղ-V      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�u   max       QW�      �  |   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �o   max       =���      �  L   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @Fp��
>            effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��p��
>    max       @vk�
=p�        (<   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @Q            h  0\   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��           �  0�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       <#�
   max       >�E�      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��z   max       B/�P      �  2d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��    max       B/�_      �  34   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >(�   max       C��      �  4   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >`w   max       C���      �  4�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         S      �  5�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          S      �  6t   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          M      �  7D   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�u   max       Q�      �  8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�l�!-w   max       ?������      �  8�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �o   max       >2-      �  9�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Ǯz�H   max       @Fp��
>        :�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��=p��
    max       @vk�
=p�        B�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @Q            h  J�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���          �  K,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  K�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���҈�   max       ?��Z���     @  L�      /         %   7               ]      &   D        S   (            	            Y               &            /   #               %               *      �            O2tP�fOwN���O��O�}O��N9�O�QmP
�QW�NO�1O��PnydNHO��P��bO�7N�g�N��Oƌ
O2�N��N�1O�k�PZT�N���O&_�N���O�vWO��eM�uOG�CN��O��6PőN�޽N���N��N%�jO�w)OUh�O ԜN���N�ҩOx�ONsO��SO�N�4�N�.N>��o�ě���o%@  ;o;o;�o;��
<o<D��<T��<u<u<�o<�o<�o<�t�<�1<ě�<�/<�h<�<��=+=\)=\)=\)=�w=�w=,1=0 �=0 �=D��=P�`=T��=T��=Y�=Y�=aG�=u=�+=�+=�+=�O�=��=��P=���=��
=�1=�-=�-=�������������������������������������������#/<HalmqqnlaUH/,�
#/3661/%#
  "*6>DGGD;/	 #/Hny��}znH</��������������������������������������������������������
&(0:UbrwvonkbI0
���5DS[_uq[)�����)'/<HD<<8/))))))))))��������
�����ZR[t��������������gZ��������������������a^]`eeht���������tka+/9Ng��������gNB2,)+BOSXat����������nh[B"#'/6<HIJH=<//+)#"" )+.+)#    ���������������������������������������������

�������245BN[gigg[YNB752222}z|����������������}������)5BMNJ5)����)54.))���""���)11.)'������

�������������#&"
�������������������������#/7;;:=HYUH</%)6BDOOQOIB63)%����� ���������������)6@DENVUGB5)��MOPSX[`hntuutthg[OMM|}������������||||||�������������������������������������������������������������������#&&!������)-46<6)����������������������#*46ACGC765*	
 #&+4:==<6/#	�����������������������������
��������������

�����)68>63)EFO[hih`[OEEEEEEEEEE),2)�������������ĺĺ��������������������������Ϲ޹��������蹻���������������������������������������������������|�w�t�v�z���(�5�A�B�J�N�S�R�N�A�5�(�&�����'�(�(�	��/�;�T�a�c�e�a�H�;�/�"��	��������	��������� �������������������������"�/�H�Y�^�]�V�I�A�/�"���������	�"�����������������������������������������׾�	��$�$�&�"���	�����׾ʾľþȾ׼'�1�@�Y�w��������f�Y�@�'�������'����'��������Ǝ�C�*�����6�X�hƁ�����m�y�����y�m�`�`�`�b�m�m�m�m�m�m�m�m�m�m���ûܻ����	���ܻ������x�r�q�����������2�N�N�H�<�7�-�/�	�������������������a�b�n�o�n�l�f�a�]�Z�X�a�a�a�a�a�a�a�a�a����(�4�A�R�Y�P�A�4������������)�6�B�T�[�R�B�6�������üöù��������)�'�)�3�L�Y�e�g�a�Y�@�3�'��������%�'��"�,�"�����	���������������	���¦§ª­©¦�|¦¦¦¦�����нݽ����"�'����ݽĽ��������������
��#�/�<�H�U�^�a�e�a�U�H�3�/�#���	�
�B�O�[�h�t�{āČā�|�t�q�h�[�Y�O�G�B�A�B�Ľнٽӽ׽׽ҽнĽý������������ĽĽĽ�ù������������������������ùëìéííù��������������� ���������������{�u�{�����5�A�N�Z�_�d�Z�N�A�5�5�2�5�5�5�5�5�5�5�5�.�;�G�R�T�]�T�G�>�.�"��	��������"�.�y�}�������������������|�y�o�o�x�y�y�y�y�����	�����	����׾������������ʾ��)�5�B�N�R�[�V�V�N�G�B�5�)��	�����)ÓÓàáãàÝÓÐÐÐÒÓÓÓÓÓÓÓÓ�A�M�f�s������s�f�M�A�4�'���� ��(�A��������������������������нĽ����������y�n�`�k�o�y�������ĽҽϽп��ĿӿпɿĿ������}�y�m�T�P�M�P�g�{�����f�o�r�����������������r�r�f�f�]�_�f�f�<�H�N�U�]�U�T�H�<�/�.�.�/�8�<�<�<�<�<�<�4�6�A�M�Q�Z�[�Z�S�M�A�6�4�0�*�(�#�(�/�4�������������������������������������������ɺֺ� �������ֺɺ����������������~���������������������~�r�e�b�f�h�k�r�~�/�H�I�P�U�a�b�h�e�a�U�N�H�E�<�7�/�'�)�/�����������������������������z�v���������y�����������������y�s�m�m�m�o�w�y�y�y�yE�E�E�E�E�E�E�E�E�E�E�E�EuEiEcE\EYEaElE�ĳĴĿ��������������ĿĳĨĦĞğĦıĳĳD�D�D�D�D�D�D�D�D�D�D�D�D�D�DpDfDlD{D�D��/�<�H�Q�U�a�b�d�a�W�U�<�/�,�#���#�#�/��� �'�)�+�(�'����
��	��������'�.�0�)�'�%��������������
������
��
�
�
�
�
�
�
�
�
�
�
�
 ( )  / 9 H L P 9 E N ) Z - � 5  ( m 3 d q p Y  + 1 I 3 V 6 � � Z + C E / \ 0 1 < d Q B n + 4 $ U \ z  }  `  �    �  i  F  d    �  	�  ]  x  �  �    �  �  �  �    �      -  �  �  �    w  �  �  [  .  �  o  �  �    1  �  �  �  �  �  \  "  0  S  �  k  X<u=#�
<�/<D��=�w=ix�<�9X<#�
<���='�=�
=<���=Y�=���<�=t�>�E�=y�#<�`B=\)=m�h=��=49X=t�=��=��=H�9=Y�=T��=�t�=��T=@�=u=�O�=ȴ9=� �=m�h=�7L=�+=��=��`=���=�{=��P=��
=�=���>bM�=�/=���=�j>%B!�KB}�B-�B�"A��zB�B�vB�B!�B&�BB �B�B"�_B2�B�lB�mB	,=B�BkB�B"�?B!�B�7B)�BLB�nB�B0�B��BN�B�KB!��B#�B��B+�QBuBBv�BRB�EB��B��B�LBwTB��B/�PB�lB��B��B B�4B�B?�B!�B��B=�B��A�� BįB7B�NB!=�B'B�B��Bb�B"�DB
U�BE�Ba�B	==B�FB�IB?_B"�B�B?3B��B|\B��B�B,�B��B=�B>�B!�TBM]BßB+�BI�B|B?�B?�B�B=�B��BʡB�OB/�_B�dB��B�!B�B��B%iB8�@�>(�A��A��1A�GQA҃�A��TAH��AWݧ@ՋaB#*Al�@���A�H
A�[A3`�A��V?��A��3A�rQA)
NA��A�/�A'�`A�I"A� �A�8WA`�Ao��ATU�A�etA�4A=��@Y;1A~�Ap�a@�VA��WA:��A���@;"@">A���A�@�Am��C��A��C��wA��@ı;@�nA�'s@�>`wA��}A��A�}BAҀ�A��WAH�eAXpN@�<B�%Al�j@�a�A���A�QA3�Aӧ?��/A���A��}A+�A��}A�|�A'�A�/A�$A�~bA_(�Ap�&AW�A���A�X�A>z@[JdA�mAs�'@�ɽA�cA;�A�gr@7�;@�wAĘ:A���AmU�C���A��C��Aï�@��r@�yA��      /         &   8               ]      '   D        S   )      	      	            Z               '            /   #               %               *      �                  '            %   %         +   S      '   3         1   !         %               +                           $   '               !                                       #               !         +   M         -                     %               '                              %                                                N���O�(O<GZN���O�`�N�B1O�`xN9�OG%	P
�Q�NO�1O��PCfNHO�O�g�O��"N�g�N��Oƌ
O2�N��N�1O�ڂP��N���O&N���O�vWO��eM�uO��N�T�O�1Oڤ`N�޽NvaxN��N%�jOt�7OUh�O ԜN���N�ҩOx�N�y�O�6O�N}&N�.N>��  �  �    �  C  X  r  �  
  4  5  �  z  �  �  �     �  2  ]  �  �  �  �  �  	/  �  �  �  �  `  �  E  3    �  �  |    �  ]  �  Z  �  �  �  �  �  �  Q    n�o%@  <#�
%@  ;�`B<��;ě�;��
<#�
<D��<�9X<u=C�<ě�<�o<�9X>2-<�/<ě�<�/<�h<�<��=+=�P=aG�=\)=#�
=�w=,1=0 �=0 �=H�9=]/=�o=e`B=Y�=]/=aG�=u=���=�+=�+=�O�=��=��P=��T>   =�1=�9X=�-=�������������������������������������������,))./<HTU`adaaXUH</,�
#/3661/%#
	#/2;ADB>;/"&&'/3<@HOUXURH=<:/&&���������������������������������������������������������
&(0:UbrwvonkbI0
�����)BV\kp[)
����)'/<HD<<8/))))))))))��������������������ZV^t�������������g^Z��������������������fdchlt���������trhff@>@FN[gt������tg[NE@ZZ]ft�����������tnaZ"#'/6<HIJH=<//+)#"" )+.+)#    ���������������������������������������������

�������245BN[gigg[YNB752222�}����������������������)8?B@.)����)54.))���!!����)11.)'������

�������������#&"
�������������������������!#/6:::8<=E</(!#)6@BLNFB86)(���������������������� )5=AACHPB5)�MOPSX[`hntuutthg[OMM~~������������~~~~~~�������������������������������������������������������������������#&&!������)-46<6)����������������������#*46ACGC765*	
 #&+4:==<6/#	����������������������������


��������������

�����)36:60)EFO[hih`[OEEEEEEEEEE),2)�������������������������������������������Ϲܹ��������ܹϹ������������������������������������������������������������(�5�A�B�J�N�S�R�N�A�5�(�&�����'�(�(��"�/�;�T�_�a�O�H�;�/�"��	�������	�����������
�������������������������"�/�P�X�W�R�H�@�?�/�"���� ���������������������������������������������׾��	����"�������׾ʾȾǾʾϾ׼'�1�@�Y�w��������f�Y�@�'�������'Ƨ������������Ǝ�\�6����6�`�uƁƧ�m�y�����y�m�`�`�`�b�m�m�m�m�m�m�m�m�m�m�������ûͻлܻۻлû�����������������������/�F�I�I�D�5�+����������������������a�b�n�o�n�l�f�a�]�Z�X�a�a�a�a�a�a�a�a�a����#�(�2�1�(�(������������� �������)�2�9�<�8�,���������������������3�@�L�Y�c�d�^�Y�L�;�3�'�������'�3��"�,�"�����	���������������	���¦§ª­©¦�|¦¦¦¦�����нݽ����"�'����ݽĽ��������������
��#�/�<�H�U�^�a�e�a�U�H�3�/�#���	�
�B�O�[�h�t�{āČā�|�t�q�h�[�Y�O�G�B�A�B�Ľнٽӽ׽׽ҽнĽý������������ĽĽĽ�ù������������������������ùìíëðñù�����������������������������������������5�A�N�Z�_�d�Z�N�A�5�5�2�5�5�5�5�5�5�5�5�/�;�G�P�T�Y�T�G�9�.�"��	����	��"�.�/�y�}�������������������|�y�o�o�x�y�y�y�y�����	�����	����׾������������ʾ��)�5�B�N�R�[�V�V�N�G�B�5�)��	�����)ÓÓàáãàÝÓÐÐÐÒÓÓÓÓÓÓÓÓ�4�A�M�Z�s�����s�f�Z�M�A�8�4�(�"��(�4����������������������������������Žý������������y�q�j�l�q�y�������Ŀɿ̿ƿ����������y�m�X�R�T�`�k�����f�o�r�����������������r�r�f�f�]�_�f�f�<�H�L�U�[�U�R�H�<�/�/�/�/�9�<�<�<�<�<�<�4�6�A�M�Q�Z�[�Z�S�M�A�6�4�0�*�(�#�(�/�4�������������������������������������������ɺֺ�����������ֺɺƺ������������~���������������������~�r�e�b�f�h�k�r�~�/�H�I�P�U�a�b�h�e�a�U�N�H�E�<�7�/�'�)�/�����������������������������z�v���������y�����������������y�s�m�m�m�o�w�y�y�y�yE�E�E�E�E�E�E�E�E�E�E�E�EuEiEcE\EYEaElE�ĦĳĿ����������ĿĳĭĦġģĦĦĦĦĦĦD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D|D�D�D�D��/�<�H�Q�U�a�b�d�a�W�U�<�/�,�#���#�#�/����'�(�*�'�"������
��������'�.�0�)�'�%��������������
������
��
�
�
�
�
�
�
�
�
�
�
�
 ,   / 9 $ K P B E K ) 0 ' �   & m 3 d q p Y  & 1 H 3 V 6 � { O ) C E , \ 0 1 < d Q B n $ + $ O \ z  �  �  �       �  �  d  �  �  �  ]  O  7  �  ,  �    �  �    �      
  �  �  ]    w  �  �  �  �    �  �      1  �  �  �  �  �  \  �  6  S  �  k  X  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  d  u  �  �  �  �  �  �  �  �  ~  p  ^  I  1    �  O  �  �  Q  �  �  �  �  x  N    �  �  W  
  �  g    �    c    }  >  m  �  �  �  �  �      �  �  �  �  ]  (  �  �  <  �    �  �  {  c  I  /    �  �  �  �  �  {  f  R  6  	  �    �    2  A  C  ;  (    �  �  �  �  ^  .  �  �  0  �  (  �  �  �  ]  �    b  �  �    *  J  X  G    �  �  &  �  �    �  g  o  r  r  j  ]  I  <  -    �  �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �    g  U  F  7  (    �  �  z  �  �      
  
       �  �  �  �  �  �  �  t  P  *   �   �  4  (      �  �  �  �  �  �  �  �  �  �  �  �  j  ?  �  �    -  5  ,  "  "  +  0  &    �  �  �  y    �  �  �    �  �  �  �  �  z  q  g  ^  T  H  =  4  (            �  �  �  �  �        $  )  3  K  t  `  E    �  f  �  {  �  1  i  �  �  �  �  �  m  G  #          �  �  k  �  x  �  O  �  �  �  �  �  �  k  W  <       �  �  �  �  r  S  D  <  4    �  Z  p  �  �  �    u  d  K  &  �  �  �  ]  $  �  �  z  �    �    P  F    �  �  �  �  9  q  y  ,  s  $  -  �  �  �    �  �  �  x  d  M  D  7  $    �  �  Q  �  U  �  >  �  2  2  1  0  0  ,  !    
  �  �  �  �  �  �  �  �  �    (  ]  Z  V  P  G  >  2  %        �  �  �  �  �  �  �  b  ,  �  �  �  �  �  n  V  5    �  �  �  u  F    �  M  �  L   �  �  �  �  �  y  o  �  �  �  �  �  �  �  s  \  C  *    �  �  �  _  8    �  �  u  d  J  ,    �  �  �  �  �  U  (  �  �  �  �  �  �  �  �  u  o  l  i  f  c  `  W  C  .       �   �  �  �  �  �  �  |  g  K  )    �  �  P  �  �  A  �  �  :  �  �  �  �  	  	.  	'  		  �  �  l  -  �  �  R  �  n  �    4  �  �  �  �  �  �  z  u  t  `  Q  B  2    
  �  �  �  �  y  ]  �  �  �  �  �  ~  o  ^  J  2    �  �  �  �  a  9  �  �  >  �  �  �  �  �  �  �  �  �  m  T  <  #    �  �  M    �  y  �  �  �  �  �  �  �  �  �  ]  +  �  �  �  B  �  �  i  !  �  `  T  >     �  �  �  X  !  �  �  w  3  �  �  $  �  �  V  �  �  �  �  z  l  c  d  f  h  i  p  }  �  �  �  �  �  �  �  �  &  :  =  (    �  �  �  �  �  �  �  x  r  �  �  u  �  �  �    *  .  2  (      �  �  �  �  z  [  <    �  �  V  �  j  r  �  �  �          �  �  �  l  0  �  �    �  �  A  @  �  �  �  �  �  �  z  Z  >  %    �  �  �  k    �  P  �  �  �  �  �  �  y  q  i  a  S  >  (      �  �  �  �  �  �  �  x  {  z  u  i  P  2    �  �  �  �  e  9  
  �  �  o  G  <    �  �  �  �  �  �  �  �  ~  [  2    �  �  y  H    �  �  �  �  �  �  �  x  b  L  7  "    �  �  �  �  �  �  w  Y  <  '  ;  F  Q  Y  ]  [  O  6    �  �  u    �  K  �  J  �  �  �  �  r  a  O  :  $      �  �  �  �  |  R  #  �  �  e  �  Z  I  3  "  
  �  �  �  �  �  �  �  q  0  �  �  5  �  x  �  �  �  �  �  �  �    s  e  V  G  8  3  6  9  <  7  1  *  $  �  �  �  �  �  k  U  >  '    �  �  �  �  �  �  �  �  �  �  �  x  S    
�  
�  
6  	�  	�  	  �    |  �  [  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  K    �  v  (  �  �  )  �  _  �    �  �  E  �  �  s  L  	  �  ?  �  �  �  [  j  "  	a  C  �  �  �  t  Y  4    �  �  i  #  �  �  >  �  �  C  �  q  �  E  M  Q  P  E  8  '    �  �  �  �  Y  +  �  �  �  c  ,  �       �  �  �  �  �  �  �  �  �  �  �  �  w  j  c  _  Z  U  n  T  :      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �