CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�"��`A�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�8�   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �y�#   max       >�u      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���R   max       @F%�Q�     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @v;��Q�     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @Q�           p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�2        max       @���          �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �q��   max       >fff      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B0�      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�|1   max       B0Q�      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�"   max       C���      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =��   max       C���      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          j      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�8�   max       P�Sj      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��<64   max       ?�s����      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �y�#   max       >�-      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���R   max       @F%�Q�     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @v;��Q�     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @Q�           p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�2        max       @��           �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Dr   max         Dr      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��PH�   max       ?�c�	�     �  Pl         Y   $               	            
             X             -      E      V   V         R   &   
                     F   3   
            $      :         %   3      M   [   j      M�8�O)Y�P���PiRmO '�Np>�N3޼N���NչvN�-O�QN���O	+mO��OQO�v�PYR�N��RO�A7N7DO�Y8N�E�P݉OI��P6X<P�f�OL6OT�P��O�MxO�Ow�N:�"N�OM��N|�N�9O���O׊�NT�pN޷�N;�jNhI�Oe�N_�vO�~N��O �eO{O�N��zO��P"�P`[N�!N�q��y�#�����`B�ě�;�`B;�`B<t�<#�
<#�
<#�
<49X<49X<D��<T��<�C�<�t�<ě�<ě�<ě�<���<�`B<�`B<�h<�h<��=o=o=+=C�=C�=�P=�P=D��=H�9=H�9=Y�=]/=y�#=�hs=��P=���=���=���=��T=��=� �=� �=\=ě�=�`B=�l�=�x�=��=�>I�>�u`^\binonnb``````````KIDN[gpt����utg[TNNK������5N[f[B����BCO[����������thOQHB%)67=BNQOGB;6)!#04<@@<0#)33)tnqz�������ztttttttta_abdnsz������zunaaa��������������������dgnk������������thhd�����

�������������$))*)(��38-*3EOht���tnk_[B63�������������������������������������;=JUnz����������zaH;�������������������������������������~��������~~~~~~~~~~����
#/HMH/#�����*/69CIMEC6*( ����	/EQ\_^XH</#�+*-/<HU^nzzoaZUHE<1+��������������������x�����
#,'�������x
#/<?AB=</#

"/7;=@?<;)"�����&.673)�����������)563$����)5;<852*)((" ! %6BHOXZdltth[O6)!����

 �������nty����������tnnnnnn��������������������#'+.#9BO[ba[ODB9999999999�����
#<?@>6/)#
�����������������������.02<>IUUIG<0........��������������������������������������56BO[`e[OB8655555555"/;HTV\]YTMH;/"/./<<GHHKH<82///////������""%#!���#%'()5BNP[`f[NB5*)##`ahmz|������zma`YX\`����$'*(#���������������������#//783/#�������	
������fbks�������������nf����������������������������������������<CEGC<0/+*+//<<<<<<<�������������������|���������������������[�g�t�w�w�w�t�q�g�[�N�H�B�A�>�?�B�N�O�[ĳ���
�<�b�u�m�b�[�f�^�#����ĺĦčąĚĳ�M������������s�M�A�4����������(�M�������������������������������x�t�n�z�������������������u�v�|������������������������������������������������āčĚĞĚĚđčĄā�u�zāāāāāāāā�m�z�����������������������z�p�m�g�l�m�m�нݽ�������������ݽ׽нŽнннп��ǿѿݿ������ѿ���������������������������������������{�u�r�f�r�����������Ź����������������ŹŰŭŠşŠţŭűŹŹ�f�������ʼʼ�����������������f�a�N�W�f���������)�4�+�+�)��������������������$�:�N�N�@�4�'������������A�Z���������������s�Z�N�5�&�����+�A����������������������������������ÿ��������������������������������������������������������������������U�a�u�{ÃÈÎÖÇ�}�q�a�H�#����#�<�U���������������~�y�m�`�_�\�^�`�m�y�z�����	��/�H�T�\�^�T�H�/�"�	���������������	�����������������������������������T�a�m�}�������������m�T�H�;��
��"�:�T������׾�	�)�-��ʾ����s�M�C�J�V�U�f��`�m�y�}�{�y�w�t�h�`�T�G�;�9�3�/�;�G�T�`�/�;�H�T�a�g�j�g�a�T�H�;�/�"�!����"�/�`�������Ŀ¿������y�m�`�R�H�A�A�C�J�T�`������5�>�7�0�+� �������ҿѿſѿ��\�h�uƁƉƃ�v�h�\�C�9�6�*���*�C�O�T�\���������ɺֺ޺ֺɺ����������������u�t������� �������׾ʾ¾ʾ׾������㹪�������¹ùʹù������������������������T�`�t�y�{�r�m�`�T�G�.�"��%�.�7�>�G�O�T���!�.�:�4�.�!�������������#�%�'�)�#������#�#�#�#�#�#�#�#�#�#������*�0�;�@�5�)�����������������������������ǽǽ����������y�p�a�`�]�`�n���ֺݺ��������׺ֺʺѺֺֺֺֺֺֺֺ��	����	����������������������������	���������������������|�x��������������������������ݿٿؿٿݿ����������#�.�5�=�C�?�<�0�#���
��������
��#�o�{��|�{�o�f�b�V�N�N�V�b�n�o�o�o�o�o�o��������)�5�>�A�6�)�����������������B�O�[�h�h�n�k�h�h�_�[�O�M�L�L�H�B�A�B�B������������������ŹŴŭšŜşŠŭŹ�����I�V�b�m�n�q�f�V�I�=�0�$���!�$�1�=�F�IDoD{D�D�D�D�D�D�D�D�D�D�D{DoDmDiDeDiDoDo�����
�� ���
������������������������¿�����
���!���
������¿¬²¿�ֺ��!�:�C�R�W�X�M�:�-�!��������ּ̺��ʼ���.�:�F�:�/�� ��ʼ������z�t�������������ûŻûû����������x�w�x�������6�*����������*�6�7�6�6�6�6�6�6�6 h 3 @ ; ? > f 7 A p ' @ & R 6 E # < V 5 k H  D 6 f 7 A " 7 u ^ � � ` . � > - ^ ^ f c N ` ? f * +   M G [ h m J    p  �  �  %  �  �    �  �  /  �  (  �  Y  �  y  #  i  W  �  �  �  �  
  Z  �  �  m  �  �  U  e  �  �  �  �  �  �  �  =  {  �  �  r    P  Z  �  Y  �  @  V  t  �  ��q�����
=��<�<�C�<49X<D��<�t�<���<�o<�h<�1<�1=49X=�P=H�9=�`B=�w=e`B=o=�t�=+=ȴ9=u===L��=Y�=�h=�hs=@�=e`B=P�`=e`B=q��=ix�=ix�>�=���=�1=�1=��=���==�Q�>t�=���=�>1'>%�T>$�>C��>T��>fff>��>+B'�B	64B�B<rBu�B%�8BMyB �B��B#NB�B#X`B�B�4B�B"a(Bl�BQ�BS�B9�B��B0�BGB�BqbBƏB��A���Bc�B�<B�#B�B$84B��B!pVB$�9B��B�B+��B&bLBFBB�A�B��B�YB�A���B�Bl)B��B2�B�B��B,�MB�FB'�B	GB��B=�B�mB%�2Be�B >zB>�B#?�B}%B#f�BO}B��B?sB!ߴB=B��B?ZB;vBPB0Q�B?=BɞB>�B	bBC�A�|1BC�B�B�CB��B$SXB9�B!B$��B-]B�B+��B&�eBAaB
B�uA�g�B�uBDcBb�A�q�B��B�B1BCBn�B�IB,�WB�$@�A���A�r�A>ު@�r@�RA�[�A�TA�nA-w�Aw�(@�A��5@쿝A�P�@�udA��A��A� �@��XA��`Al��A�hA�4�A���AP��Ag�nA��~Am7A��OBÀ@�EAU�>�"Ae��A��A��tA��A�@A�fA�*{A���A~E�A�0mBmwA�fAٷ�A��B�C���A���A�	�@^�A ��@�S6A�r<@��(A�ShA��AAp@���@�2�A�u�Aݤ4A�<A.�oAv��@��A�.�@�m�Aԍ@��A��mA��A��@��KA�IRAk)=A���AҀ�A��AQ��Ai �A�ywAl��A�&�B3�@$	+ASŵ=��Ah2�A^�A���A���A��@DEA��IA���A}�vA�}RBAAӉzAڊ�A�[�B
�hC���A�A���@o&`A ��@�5�A�x�         Z   %            	   
            
              Y      !      -      F       W   V         S   &                        F   3      	         %   	   ;         &   3      N   [   j               C   5                     %         '      !   /            )      %      +   A         %   !                           #                                       %   -   1               ;   '                     %         '      !   '                  !         !                                       #                                       #   -         M�8�O�P�SjPVN�<Np>�N3޼N���N�?N�-O�QNH��O	+mO��OQO�v�P)�N�O\��N7DO��N�E�O�N�`XO�QLO��OL6OT�O~3O~Y�N���Ow�N:�"N�OM��N|�N�9O��^O�f,NT�pN޷�N;�jNhI�Oe�N_�vOh!�N��O �eO{O�N��zO�.*P"�O�g�N�!N�q�  n  �  �  V  %    Y  �  �  Q  �  �  �  !  �  H  	G  �  �    �  
  �  ^  
  -  ,  T  
e  [  b      N  X  @  _  
�  �  �  �    �  
T  �     �  �  �  7  �  �  {  L  �  x�y�#��t�<u;��
<o;�`B<t�<#�
<49X<#�
<49X<e`B<D��<T��<�C�<�t�=0 �<���<�h<���=t�<�`B=49X='�=�C�=�\)=o=+=�7L=,1=��=�P=D��=H�9=H�9=Y�=]/=���=���=��P=���=���=���=��T=��=�^5=� �=\=ě�=�`B=�l�=�h=��>�->I�>�u`^\binonnb``````````MKLN[gmt����tsg[WPM�����)5N[[OB�����KJO]h����������th[OK &)56<BKB@:6) !#04<@@<0#)33)tnqz�������zttttttttebdenuz������zsneeee��������������������dgnk������������thhd�����	����������������$))*)(��38-*3EOht���tnk_[B63�������������������������������������EDMUaz���������znaKE������������������������������ ���������~��������~~~~~~~~~~�������
"#����*/69CIMEC6*( ��#/<HTXYVH</#
205<HLU\VUH<22222222��������������������������
�������
#/<?AB=</#

"/7;=@?<;)"������%(("�����������'//*��")5:;975.)'#!! %6BHOXZdltth[O6)!����

 �������nty����������tnnnnnn��������������������#'+.#9BO[ba[ODB9999999999�����
#/2541+#
�����������������������.02<>IUUIG<0........��������������������������������������56BO[`e[OB8655555555"/;HTV\]YTMH;/"/./<<GHHKH<82///////������
#! ��#%'()5BNP[`f[NB5*)##`ahmz|������zma`YX\`����$'*(#���������������������#//783/#�������
 ������fbks�������������nf����������������������������������������<CEGC<0/+*+//<<<<<<<�������������������|���������������������[�g�t�t�v�v�t�p�g�e�[�N�I�B�B�?�@�B�N�[����<�U�[�W�I�E�F�#�������ĻĞĘĳ����M�f���������������s�Z�A������=�M���������������������������x�u�o�x�|���������������������u�v�|������������������������������������������������āčĚĞĚĚđčĄā�u�zāāāāāāāā�m�z���������������������z�q�m�i�m�m�m�m�нݽ�������������ݽ׽нŽнннп��ǿѿݿ������ѿ�����������������������������������z�u���������������������Ź����������������ŹŰŭŠşŠţŭűŹŹ�f�������ʼʼ�����������������f�a�N�W�f���������)�4�+�+�)��������������������$�:�N�N�@�4�'������������A�Z�s������������s�Z�N�1�"����!�5�A��������������������������������������������������������������������������������������������������������<�H�U�a�n�t�x�~À�z�n�H�<�#�����#�<���������������~�y�m�`�_�\�^�`�m�y�z������/�;�H�S�V�U�N�@�/�"��	�����������
�������
�������������������������������T�a�m�y�����������z�m�a�T�D�;�;�<�C�P�T�����ʾھ��������ʾ������������������`�m�y�}�{�y�w�t�h�`�T�G�;�9�3�/�;�G�T�`�/�;�H�T�a�g�j�g�a�T�H�;�/�"�!����"�/�m�y�����������������y�m�`�[�R�Q�V�`�f�m�������(�+�.�+�&��������ݿܿ޿��\�h�uƁƄƁƀ�t�h�\�O�C�<�6�/�6�C�O�V�\���������ɺֺ޺ֺɺ����������������u�t������� �������׾ʾ¾ʾ׾������㹪�������¹ùʹù������������������������T�`�t�y�{�r�m�`�T�G�.�"��%�.�7�>�G�O�T���!�.�:�4�.�!�������������#�%�'�)�#������#�#�#�#�#�#�#�#�#�#������#�*�/�2�,�)����������������������������ĽĽ��������y�r�d�d�^�`�k�w���ֺݺ��������׺ֺʺѺֺֺֺֺֺֺֺ��	����	����������������������������	���������������������|�x��������������������������ݿٿؿٿݿ����������#�.�5�=�C�?�<�0�#���
��������
��#�o�{��|�{�o�f�b�V�N�N�V�b�n�o�o�o�o�o�o��������)�3�6�:�8�6�����������������B�O�[�h�h�n�k�h�h�_�[�O�M�L�L�H�B�A�B�B������������������ŹŴŭšŜşŠŭŹ�����I�V�b�m�n�q�f�V�I�=�0�$���!�$�1�=�F�IDoD{D�D�D�D�D�D�D�D�D�D�D{DoDmDiDeDiDoDo�����
�� ���
������������������������¿�����
��� ���
������¿­²¿�ֺ��!�:�C�R�W�X�M�:�-�!��������ּ̺����ʼּ����	� ���ּʼ������������������������ûŻûû����������x�w�x�������6�*����������*�6�7�6�6�6�6�6�6�6 h . C I 0 > f 7 > p ' T & R 6 E   @ M 5 = H ! /  K 7 A   ] ^ � � ` . � + * ^ ^ f c N ` = f * +   M G [ 4 m J    Q  �  �  �  �  �    �  �  /  {  (  �  Y  �  �    �  W  m  �  �  �  	  �  �  �  �  �  I  U  e  �  �  �  �    �  �  =  {  �  �  r  �  P  Z  �  Y  �  /  V  %  �  �  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  n  k  h  e  b  _  \  Y  W  T  U  Y  ^  b  g  l  p  u  y  ~  �  �  �  �  �  �  �  �  s  W  9    �  �  �  �  e  5    �  q  �  2  k  �  �  �  v  S  )  �  �    �    x  �  �  �  �  �      5  G  R  U  O  C  0    �  �  �  �  ~  V  "  �  !       $      �  �  �  �  �  �  q  T  9  .  )  *  ,  I  i       �  �  �  �  �  �  �  �  �  �  �  �    x  v  s  p  m  Y  T  O  I  D  >  9  2  *  "          �  �  �  �  �  �  �  �  v  c  P  >  -        �  �  �  �  �         %  8  �  �  �  �  �  �  �  �  �  �  �  t  b  P  ;  %    �  �  v  Q  K  E  @  7  .  %    �  �  �  �  �  s  X  =  "     �   �  �  �  }  |  ~  y  n  b  X  L  :  "    �  �  �  x  ?  �  �  p  y  �  �  �  �  �  �  �  �  �  �  ~  c  D  "  �  �  �  C  �  �  �  �  �  �  �  v  a  J  2    �  �  �  �  �  �      !    �  �  �  �  �  v  X  F  m  [  G  (  �  �  Z       �  �  �  �  �  �  �  `  4    �  �  o  <    �  t    �  4  �  H  ;  0  %        �  �  �  �  �  �  �  �  �  w  2  �  �  �  �  	$  	=  	F  	?  	'  	  �  �  R  �  �  5  �  5  |  }  6  �  �  �  �  �  �  �  �  �  t  \  A    �  �  `  �  �  C   �   z  �  �  �  �  �  �  �  �  �  i  >    �  �  S    �    �  �    	    �  �  �  �  �  �  �  �  �  �  �  �  �           )  *  �  �  �  �  �  �  �  a  6    �  �  P  �  y    �  �  
  �  �  �  �  �  �  �  �  �  �  o  ]  I  5  "     �   �   �  $  W  �  �  �  �  �  |  `  9     �  |  ,  �  ]  �  �  )  [    ]  �  �    1  L  \  [  Q  ?  "  �  �  l    �  P  �  �  5  �  �  �  	G  	�  	�  
  
  
  	�  	�  	�  	`  	  �  �  �  �  �  �    H  t  r  Q  6  0  '  *    �  �  �  T  �  H  �  �    ,    �  �  �  �    ^  4    �  �  u  P  6    �  �  W  �  T  M  @  #    �  �  �  ]  (  �  �  w  -  �  �  j    �    �  	0  	�  	�  
  
8  
R  
b  
d  
Y  
3  	�  	�  	`  �      �  Y  +  �    >  R  [  W  L  ;     �  �  �  ]    �  M  �  (  �  `  N  Y  a  Z  Q  C  5  $    �  �  �  �  �  �  �  F  �  �  d    �    �  �  �  �  �  s  Y  a  n  t  |  h  B                    �  �  �  �  �  �  �  �  �  �  �  �  �  y  l  `  N  �  �  �  �    �  �  �  U    �  �  b  /    �  �  �  l  X  J  ;  )      �  �  �  �  �    e  E    �  �  �  \  &  @  4  (        �  �  �  �  �  �  �  �    r  d  W  J  =  _  Y  S  M  G  A  ;  5  1  ,  '  "                
  
  
�  
�  
�  
�  
�  
�  
�  
`  
  	�  	d  �  �  �  Y  �  �    �  �  �  �  |  _          �  �  �  �  A  �  �  �  h  �  �  �  �  �  �  �  �  �  �    t  h  [  I  5  "    �  �  ~  6  �  �  �  �  �    h  Q  2    �  �  �  z  J  /    �  �  v    
  �  �  �  �  �  �  �  �  n  Y  D    �  �  I    �  �  �  �  �  �  �  w  l  b  X  M  B  7  +        �  �  �  �  
T  
?  
  	�  	�  	c  	7  �  �  g  �  �  !  �  x    �  f  �  �  �  s  U  ?  -      �  �  �  �  �  z  L    �  �  �  P            �  �  �  W    �    
�  	�  	H  �  �    &    �  �  �  �  w  ]  =    �  �  �  p  =  	  �  �  o  1  �  �  7  �  �  �  {  X  )  �  �  �  a  %  �  �  d  "  �  �  e  �  �  �  �  �  ^  B  $  �  �  �  ]     �  �  2  �  0  �  �    �  7  �  �    P    �  �  3  
�  
V  	�  	4  �  �    ,      �  �  �  q  `  @    �  �  �  �  �  ]  '  �  �  �  \    �    �  �  �  �  e  :     �  b  �  �  
�  
P  	�  �  �  1  D  \  ]  {  @    �  �  �  ~  _  -  
�  
�  
]  	�  	�  	  o  �  �  <  d  B  �  �  #  )  7  L  G  8    �  �  +  
�  
"  	h  �  a  �  a  �  �  �  �  �  l  M  +    �  �  O  �  �  M  �  �  ;  �  �  x    �  �  �  y  Q  &  �  �  �  f  '  �  c  �  =  �    y