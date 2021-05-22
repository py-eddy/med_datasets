CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��t�k      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Nm&   max       P��9      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �o   max       =��-      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�ffffg   max       @D�z�G�     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(��   max       @v�fffff     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q            l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @�@�          �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       >F��      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B&��      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��g   max       B&@�      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��4   max       C�ޛ      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?���   max       C��      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          '      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M͇   max       P(      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�&���   max       ?�;�5�Xz      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       =�
=      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�ffffg   max       @D���R     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v�fffff     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P@           l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @�@          �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��_o�   max       ?�:)�y��     �  M�                           
                           )   E      	      b   @   �      	               1      
   7            	            2   %                  ?         O(�	O8��NҥNGl�Oo�?OZmN��fN	_�N�/�OspN��KNAyPO|�#O͕rNZ�Nm&O�<xP�`�P%-�N)ZNn ;O"6P� P��9P�U|N��XN�hHNqrN�O=ONbN�P	O+B�O�O���N�N���N��:Nv<FN�K�N�%�N�5�Or	�O��\O�*lO��N��N�@�NM�O��O���N^QN1�<�o�ě���9X�T���ě�%   ;o;o;ě�<t�<49X<D��<D��<�o<�C�<�C�<�t�<���<���<�1<�1<�1<�9X<�9X<�9X<ě�<���<���<���<�<��=o=o=o=+=��=��=��=#�
=#�
=@�=@�=L��=e`B=q��=q��=�+=�7L=�C�=�C�=�\)=�t�=��-��������������������fc^^g���������wtrjgf����������������������������������������LOJN[gt��������tg[NL|������������������|���������������������������������������moz�������������zxom��������������������{�����������������{{��������������������(()),18<>INU^_YUI<0(~������������������~��������������������)*)$ 
-/<CGHKKA/#� +-N[�����gFCKB)������
#--%!
�����������������������������������'&)1BO[^^\[POEB86*)'����5BNXZU>::5)�����5BLO[aed`NB�"$)5N��������g[)!"/685/"COZ[d[XOB6)))-,6?CCChmnz��{zvnhhhhhhhhhh !)+)	        ##&)/<HMUXVUQH</*###;:<EHU[`VUPHF<;;;;;;rqz���������������zr����������������������������������������#/0<BFJKJG</#������������������������������������������
#%/06/,#
�����������������������			


		������������������������������������������������

����������	����"/;HQTXZXTQH;/��������������������[[cht|ytihh[[[[[[[[[��������������������������������)6=86'""�����������������������mt{�������vtmmmmmmmm#$+*#���'������������������������	��a�n�zÇÑÏÇ�z�s�n�a�[�U�M�H�F�H�U�^�a�<�I�L�M�I�<�0�+�0�9�<�<�<�<�<�<�<�<�<�<�{ŇŉŋŇ�{�{�u�n�b�^�b�n�q�{�{�{�{�{�{����������������������������������������āččĚĦĩĦĢġĚčĊā�x�t�p�m�t�yā�Z�f�s�w�������{�s�f�f�`�[�Z�V�Z�Z�Z�Z�'�3�8�4�3�0�'�$�!�$�'�'�'�'�'�'�'�'�'�'�g�s�}�������}������s�g�]�Z�N�J�M�N�Z�gÓàìùû��������ùìàÓÐÇÀÀÇÎÓ�#�/�1�<�H�O�N�H�B�<�3�/�'�#���� �#�#���������������������������������������ҼY�f�r������������ʼ��������r�a�S�U�P�Y�A�Z��������������z�s�Z�A�4�(� �!�(�4�A�)�+�6�9�6�,�)�����
����)�)�)�)�)�����������������������������������������(�-�/�5�N�S�\�N�5�(������������(��������/�H�l�n������������������������5�B�[�g�p�q�g�[�����������)�5�`�a�m�y�{���y�m�h�`�`�`�`�`�`�`�`�`�`�`�_�l�q�x��x�l�_�S�S�F�:�F�K�S�[�_�_�_�_���
�������
�������������������������
�#�I�j�pōŏŉ�{�n�U�������ĳ������ƚ������$�6�I�L�=�0������ƳƉ�|�l�uƚ�)�6�[čĥĲĩ�u�h�6�)�������������)�a�k�m�z�~���z�m�a�Z�X�[�a�a�a�a�a�a�a�a������ݿԿѿɿſſпѿҿݿ�������������
�
��
�������������������������������f�f�k�i�f�_�Z�W�S�W�Z�f�f�f�f�f�f�f�f�f�����������������������������������ù����������ùìàßàäì÷ùùùùùù�������������	������ŹŪŚŝŪŦŧŵ���5�A�N�Z�_�d�d�d�a�Z�T�N�A�=�5�.�(�'�(�5���������������������s�n�g�c�d�g�j�s�u��D�EEEE7EMESEPEDE7E*EEED�D�D�D�D�D��s����������������s�g�Z�V�Z�g�r�s�s�s�s���(�5�A�H�A�;�5�(�����������m�q�w�r�n�m�a�`�T�K�G�=�>�G�M�T�`�k�m�m�z����������z�q�m�l�m�u�z�z�z�z�z�z�z�z�׾����	��"�*�%�"��	�������׾Ӿ׾׾�(�4�A�A�E�A�<�4�(���
��������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�EuEhEgEuE������������!�&� ���������������������0�9�<�;�8�0�,�#��
��������������
��0FF$F1F=FJFVF_FVFVFJFDF=F1F$FFFFFF�'�(�4�8�7�4�'�����'�'�'�'�'�'�'�'�'���	���"�#�"�"���	������������������D{D�D�D�D�D�D�D�D�D�D�D�D{DrD{D{D{D{D{D{�L�Y�r�~�������������~�Y�@�3�'�� �+�@�L�Y�f����������������������w�n�i�`�Y�U�Y�����������������������������������������"�/�;�D�=�;�/�"���"�"�"�"�"�"�"�"�"�" R 6 9 D ( , H z @ * 5 _ M L h ] > a  j d 8 X 9 , @ ; > x  \ C D $ 4 � G I , � 8 Z L .  : I & � N c = R    j  �  ,  _  �  U  �  X    #  ,  �  +    �  <  :  �  �  v  �  A  �  �  �  �    4  �  -  �  �  �  X  %  n  �  &  }  7  �  �    �  6  2  E    �  J  �  |  A����;��
��t��#�
<T��<�o<e`B;��
<u<��<�h<���=C�<���=C�<���=��=q��=�9X<�j<�<���=�=� �>F��=C�=C�<�h<�h=P�`=C�=��T=P�`='�=�-=#�
=<j=P�`=H�9=0 �=ix�=T��=���=�j=� �=���=�hs=��-=���>�=��=�1=�1B�HB	�zB0�B��B	z�B,�BkB!9�B �B!�B�xB�B&��B�B��B��B��B��B�B�`B�B>'B�B��B�)A���BKHBB�jB��Bd�B�BUBBXB!B0�Bk~BI�Be@B0B G#B"��B��B��A�#�B��B٘B�Bv�B3�B/"B�B�eB>�B	�$B=�B��B	@}B��B��B!bB �B"@�B�DB��B&@�B�kB�IBN�B��B	�B4�B�B�B>�B�JB=�B��A��gBC:B>�B>/B�lB|�B��BD-B�PBCBW�B�#BD�BFNB��B D�B"�zB�>B�jA�B�4B�qB�B�&B�eB��B�;B��A���AǕA츝A�!�A�U�A��AA� ?��4A�}�A˻EAqA�Ɲ@�MfA@��A���Ar��A�uA���A�DAj��@���A�^A�+B�YA�rA�FA|�BA��A@RA�	A�<�A���A��"A���C�q�A��{A�4AhnA��A[aA7O�A2V�C��A��A��C�ޛ@�=BA�APC��g?�+@�^@G�A��A���A�|�A�y1A�~�A�,A���AC ?���A�#�A�l�A�A��@�aAC#�AցAr��A�� A���A�k�Ak"�@�?�A�4�A�DB��AقA���A|	jA���A?�A�]~A͆�A���A��A�iCC�p!A��mA�}�Ai �A��lA\��A7�A0��C�aA��cA�}ZC��@�}�A�!�C��s?���@�y@��A�|�                           
                           )   F      
      c   A   �      
               2      
   7      	      
            3   %                  ?                                                    %         !   G   )            5   9   9                     '                                                      %   !                                                %            %               #   '                        '                                                         !      OG�N���NҥNGl�OMɽOZmN�AzN	_�N�P�OspN�>-NAyPOY�rO͕rNZ�Nm&O��PKO��N)ZNn ;O"6O�y/P(O���Nrc
N�hHNqrN�NJ>lNbN�O��YN��cN�sOfv>N�N���N��:Nv<FNR�N���N�bO��O�WO�*lO��N��N�@�M͇O���O���N^QN1�<    �  \  �  l  �  f  �  �  x  �  �  �  �  �  �    x  
  �  d  �  	�  �  �  �  �  W  3  �    �  �  �  
�  v  �  �  ,  �  6  Z  
�  �  �  7  �  �  +  u  B  �  $��󶼃o��9X�T����o%   ;D��;o;�`B<t�<�t�<D��<e`B<�o<�C�<�C�<���<��=D��<�1<�1<�1=y�#=D��=�
=<���<���<���<���=�w<��=t�=t�=C�=#�
=��=��=��=#�
='�=H�9=D��=�%=y�#=q��=q��=�+=�7L=�\)=�1=�\)=�t�=��-��������������������begt���������tnjhgbb����������������������������������������MNS[^gt������}tg[OQM|������������������|����������������������������������������qnpz����������{zqqqq������������������������������������������������������������****-03:<IU\]WUTI<0*~������������������~��������������������)*)$�
 '/<AFIJ?/#%'68N[ejr�����tNB5)%�������

�����������������������������������'&)1BO[^^\[POEB86*)'����)576/+)���� )5BJSTQLB5���47=BN[nz~xtg[N?864"#/474/"COZ[d[XOB6)))-,6?CCChmnz��{zvnhhhhhhhhhh !)+)	        .//:<>HOKH<<;/......;:<EHU[`VUPHF<;;;;;;uuz���������������zu����������������������������������������#/<?CGIHHB</#������������������������������������������
#%/06/,#
�����������������������	

						���������������������������������������������


������������
�����"/;HQTXZXTQH;/��������������������[[cht|ytihh[[[[[[[[[����������������������������������')+(�����������������������mt{�������vtmmmmmmmm#$+*#������������������������������zÇÉÊÉÇ�|�z�n�c�a�U�R�L�U�a�n�p�z�z�<�I�L�M�I�<�0�+�0�9�<�<�<�<�<�<�<�<�<�<�{ŇŉŋŇ�{�{�u�n�b�^�b�n�q�{�{�{�{�{�{����������������������������������������āččĚĦĩĦĢġĚčĊā�x�t�p�m�t�yā�Z�f�s�u�������y�s�j�f�a�\�Z�Z�Z�Z�Z�Z�'�3�8�4�3�0�'�$�!�$�'�'�'�'�'�'�'�'�'�'�Z�g�s�{�������{�s�g�`�Z�N�K�N�N�Z�Z�Z�ZÓàìùû��������ùìàÓÐÇÀÀÇÎÓ�#�/�<�E�E�>�<�/�#�#� � �#�#�#�#�#�#�#�#���������������������������������������ҼY�f�r��������������������r�h�f�Z�W�S�Y�A�Z��������������z�s�Z�A�4�(� �!�(�4�A�)�+�6�9�6�,�)�����
����)�)�)�)�)�������������������������������������������(�,�-�5�A�O�Y�N�5�����������������/�H�^�[�T�H�;�,��	�����������������5�B�N�[�a�g�i�i�g�[�N�B�)������)�5�`�a�m�y�{���y�m�h�`�`�`�`�`�`�`�`�`�`�`�_�l�q�x��x�l�_�S�S�F�:�F�K�S�[�_�_�_�_���
�������
�������������������������
�#�0�G�T�X�W�N�I�0�
���������������������������������ƳƢƘƌƒƚƧƳ���[�tāĆĈĄ�t�h�[�O�6������)�6�B�[�a�i�m�z�z���z�m�a�[�Y�\�a�a�a�a�a�a�a�a������ݿԿѿɿſſпѿҿݿ�������������
�
��
�������������������������������f�f�k�i�f�_�Z�W�S�W�Z�f�f�f�f�f�f�f�f�f���������� ��������������������������ù����������ùìàßàäì÷ùùùùùù�����������������ŹŭşŢũŬũūż���5�A�N�Z�[�`�`�[�Z�N�B�A�5�2�,�,�5�5�5�5�s�������������������s�h�h�o�s�s�s�s�s�sD�EEEE*E7EDELE?E7E*EEEED�D�D�D�D��s����������������s�g�Z�V�Z�g�r�s�s�s�s���(�5�A�H�A�;�5�(�����������m�q�w�r�n�m�a�`�T�K�G�=�>�G�M�T�`�k�m�m�z����������z�q�m�l�m�u�z�z�z�z�z�z�z�z���	��"�&�"�"��	��������������������(�4�=�A�C�A�8�4�(��������(�(�(�(�������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�ExEuEoEvE�E�E�������
����"���
�������������������0�9�<�;�8�0�,�#��
��������������
��0FF$F1F=FJFVF_FVFVFJFDF=F1F$FFFFFF�'�(�4�8�7�4�'�����'�'�'�'�'�'�'�'�'���	���"�#�"�"���	������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��@�L�Y�e�r�~�����������~�r�L�6�3�/�3�7�@�Y�f����������������������w�n�i�`�Y�U�Y�����������������������������������������"�/�;�D�=�;�/�"���"�"�"�"�"�"�"�"�"�" h 1 9 D + , E z % * 2 _ L L h ] > :  j d 8 H & - C ; > x ) \ B 2 ! . � G I , k 0 V B "  : I & v D c = R        ,  _  �  U  �  X  �  #  �  �  �    �  <    z  0  v  �  A    �  �  �    4  �  _  �  q      �  n  �  &  }  �  �  �  Z    6  2  E    B  F  �  |  A  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�                  	    �  �  �  �  �  �  �  �  �  h  �  �  �  �  �  �  �  �  �  �  _  '  �  �  P  �  �  g  \  8  \  X  U  Q  N  J  F  B  >  :  0  !      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  i  Y  H  7  %       �  S  d  l  l  f  [  G  0      �  �  �  �  �  j  8      �  �  �  �  �  �  �  r  `  G  )    �  �  �  g  �  �  �  �  �  `  d  b  X  M  @  /    �  �  �  �  �  z  R    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  F  h  �  �  }  r  f  W  G  5  "    �  �  �  �  �  r  N  )  x  j  Y  B  &    �  �  �  �  c  <      �  �  �  �  �  �    8  R  e  y  �  �  �  �  �  v  V  0    �  �  _    �    �  �  �  �  �  �  �  �  u  a  L  7       �  �  �  �  z  [  �  �  �  �  �  �  �  {  i  \  6  �  �  �  ^  n  0  �  7   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  h  Y  I  2    �  �  �  }    5    �  �  �  Z  #  �  �  r  3  �  �  g    �  �  �  �  �  �  �  �  �  �  �  {  r  j  a  X  P  G  ?  6  �    �  �  �  �  �  �  �  �  �  �  c  8    �  �  e  +    �  �  �  !  i  s  v  h  B    �  ]      �  �  �  �  o  �  �  :  �  �  �  �  �    
  �  �  �  �  G  �  �    O  �  _  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  d  �  �  �  �  �  q  _  M  :  &    �  �  �  �  n  K  '    �  �  �  �  �  {  r  i  ^  P  B  5  '    
   �   �   �   �   �  �  �  �  	  	4  	�  	�  	�  	�  	�  	�  	S  �  �  �  S  �  �  �  J  ^  p  }  �  �  �  �  �  �  �  �  z  S    �  X  �  
  M  q  L  �  `  �  �    5  ]  y  �  �  y  [    �  �  
�  	,  e  &  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  \  H  4      �  �  �  �  �  v  f  W  H  E  J  I  A  8  0  *  ,  -  +  )  W  V  U  T  R  P  I  B  <  5  .  '             �  �  �  3  -  (  "          �  �  �  �  �  �  �  r  R  2     �  �  $  J  n  �  �  �  �  �  �  �  �  q  M  '  �  �  �  �  �            �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  �  �  �  �  �  �  b  >  )    ,  )    
  �  �  .  �  �  �  �  �  �  �  �  �  �  �  �  �  �  c  4    �  �  R    �  �  m  w  �  �  �    x  n  c  S  A  +    �  �  �  �  �  �    
Q  
p  
�  
�  
{  
e  
F  
  	�  	�  	F  �  W  �  !  q  �  �    0  v  p  j  c  ]  W  Q  K  E  ?  A  L  V  `  k  u  �  �  �  �  �  �  �  �  �  �  �  �  �  z  m  ^  O  =  +        H    �  �  �  w  f  W  I  7  #    )      �  �  �  \    �  �  ,  '  #  
  �  �  �  �  �  �  �  x  c  H  %  �  %  t     �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  m  ]  M  <  ,        *  0  4  ,  "      �  �  �  �  l  F    �  �  �  s  X  Y  Y  Y  W  J  <  /       �  �  �  �  �  �  �  q  X  @  	�  
!  
h  
�  
�  
�  
�  
�  
�  
7  	�  	P  �  C  �    _  �  �  �  7  f  �  �  �  t  Z  :    �  �  e     �  �  9  �  �  p  �  �  �  �  �  p  X  9    �  �  �  |  P    �  �  $  �    S  7  3  $    �  �  �  �  �  �  g  J  +    �  �  �  q  G  E  �  �  �  �  �  q  `  O  A  8  /  &      �  �  �  �  �  �  �  �  �  �  w  d  P  8       �  �  �  �  �  g  H  )    �  �      &  (  *  /  8  @  h  �  �    S  v  i  \  N  @  1  b  =  3  8  d  t  h  M  (    �  �  N  �  �  
  d  j    B  B  /    �  �  �  a  ,  �  �  n  7    �  |    �    �    �  �  �  �  p  V  9    �  �  �  d  "  �  �  X    �  �  H  $    �  �  �  �  �  �  x  `  G  0  &      	     �  �  �