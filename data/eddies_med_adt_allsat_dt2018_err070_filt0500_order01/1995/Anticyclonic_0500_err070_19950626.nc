CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�Z�1'      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P��u      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =��      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?z�G�   max       @FU\(�     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�z    max       @vx�\)     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q            x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���          �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �ě�   max       >Kƨ      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�k   max       B1�o      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B1�v      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?j�   max       C�y�      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?P�}   max       C�mA      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P(�a      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��{���n   max       ?�n.��3      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��   max       =�`B      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?:�G�{   max       @FU\(�     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�\    max       @vx�\)     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @Q            x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�E�          �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��䎊r   max       ?��	k��     �  T                  %   	      J   V         U   ,   	      "      	   w                     %   
   	         	   &                        #         �            "         "   !         #      (   4   9   "N��@Op<O���Oك�Oz�;O��N/�gNI#�P��uP�l�O`7�OzbP�L�P�NjmBN|h�O<f%O���N� UP#�oN3RuOu�6N'5�N$�tN�H�N[�$O�O��N���N�cO��{N��O���N���OY�N��N�)�Nc}�N��O���Oyq.P?UO0e�O�՜O�k�O��dN��O\mOL�iM��Ol(O��(N��,N�`O5`�N�/O;�O�3�O���Oa�W��󶼓t���C��T���49X�D���D���o;o;o;D��;D��;ě�<o<49X<49X<D��<T��<e`B<�o<�o<�o<�t�<���<��
<��
<��
<��
<�1<�j<ě�<���<���<�/<�<�<��=o=\)=\)=t�=�P=�w='�=0 �=0 �=49X=@�=L��=L��=e`B=ix�=��=�C�=��
=��
=�-=���=���=����������������������ZXZ[egt�������{tgc[Zlihjkinz���������zsl��������
���JUXinz|������znaULHJLJEN[g��������tgcWOL��������������������Z[gtuyvtg^^[ZZZZZZZZfdz��������������zuf�����������������)/5BNZWNB?64)ytu����������������y����,,5KUUB5����5BLMLQRMMSB5)���������������������������������������!!"#/<HTU]\USH</)#!����)+-/-) ����������������������27BN[gy�������tg[M62��������������������LGIR[ht��������th[OL������������������������������������������������������������������������������������&-1483)����56BOPW[\]][WOGBB?965�����

��������LJNOQ\hosohgb\SOLLLL|}�����������������|��������������������
#0EIOQQLG<0#]_aahhht��������th]]��������������������������

�������fnnz{�����������{nff%$)5=BFJB;5)%%%%%%%%ABGLNX[egmoc_[VONHBA����������������������������������������
#218F<#���������������������������������

�����^]`amz����������zma^�������������������������������������������������������!)))59;>:5&%()+5565)&&&&&&&&&&$%),38;HT[`b_TLH=;/$�������������������������������������������������������������������������#!
�������	

������������������)6=BEEB6.))(,46<BCOT[`hf[OB63)�0�=�A�F�>�=�0�%�$� �$�'�0�0�0�0�0�0�0�0�zÇÓàìïù÷àÓÇ�z�n�e�a�]�^�a�n�zE�E�E�E�FFF$FFFE�E�E�E�E�E�E�E�E�E������)�5�B�[�[�N�B�5����������������������������������������������������������������������������������������������������������������������������������������n�p�q�p�n�b�U�R�M�U�b�k�n�n�n�n�n�n�n�n����������;�N�U�H�;�	�����������t�{������6�M�TāĘč�h�L�)������÷íö������;�H�T�a�m�s�m�i�d�^�T�H�@�3�/�"�"�)�,�;�/�<�H�N�T�Q�H�C�<�2�/�(�#�����#�$�/�<�U�n�|ŗŗŇ�b�Q�<�#�
������������#�<�"�G�T�a�r�w�m�`�T�;�.�	��׾Ⱦ¾þѾ�"�����������������������	��� �"�&�(�"����	���	�	�	�	�	�	�������������������������������������������������������������������������5�A�N�X�Z�e�]�Z�N�A�@�5�4�2�5�5�5�5�5�5�B�[�wĆĊćā�{�h�Q�B�6�#�������)�B�6�8�>�B�D�B�6�)�!�%�)�4�6�6�6�6�6�6�6�6�����������ƿ˿ʿǿĿ��������������������������ʼ˼˼ʼ�������������������������������������ֻܻܻ�������������������(�5�A�C�C�A�5�(���
���������ݿ�����������ݿܿܿݿݿݿݿݿݿݿݿ.�G�`�k�o�p�h�`�T�;�.���������	��.������׾̾ʾ������������������ʾ׾㿒��������������������������������������������	���	������׾Ծ׾ݾ����4�M�Z�_�^�Z�X�L�A�4�(��������0�4�����������������������������{���������������������������������r�f�X�Q�R�Y�f�r���:�F�S�_�l�l�l�i�b�_�S�Q�F�D�:�6�1�1�:�:������1�0�.�+�+�����
���������(�4�A�M�U�W�M�A�4�(�����������������!�!�'�!�����������������t�o�t�w�v������(�+�5�6�5�(���	�����������5�A�N�g���������������s�A�(���&�+�-�5���ʼּ�������
�����ּ������������	�;�A�?�1�.�0�"�	���������������������	�y�������������������y�l�f�`�`�Y�`�l�v�yD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�Ŕŭ����������ŻűŭŠŔōŇŇŋŊŊŐŔ�T�W�V�T�V�\�e�\�O�6���	����6�C�K�T�H�U�Y�a�i�n�v�zÇÊÊÇÅ�z�n�i�a�U�H�H�Y�f�r�����������������������f�]�N�O�YƚƧƳ����������������������ƳƧƢƑƗƚ���$�%�$����	������������ĳĿ���������������������ĿĺĳįĪĳ�������
���#�)�*�%��
���������������ؾ��&�(�4�A�M�Q�S�N�M�A�<�4�(�����ÓàìñìäàÓÇÏÓÓÓÓÓÓÓÓÓÓ������!�-�6�:�E�:�-�)�������׺��S�_�l�w�x���������������x�l�h�_�S�H�N�SE�E�E�E�E�E�E�E�E�E�E�E|EuEiEhEaEiEuE~E������'�+�3�8�:�6�3�'�������ݹڹ���e�r�~�����������������r�Y�O�L�F�@�=�T�e���'�4�8�@�L�M�Y�M�G�@�4�'������� ? , : > 0 % L 8 I < - : F ^ C { $ 9 ;  g ; K ; = 3 0 ^ i , > ; ! + C V c R � F 2 l +  I > � [ O k Q & v ` R w 2 F ; D  �  �  �    �  �  e  o  6  �  �  V    -  �  �  �    �  �  �    Y  I  �  n    ]  �  �  2  �  N     �  �    v    g  �     q  �  0  c    �  �      
  �    �  V  '  p  :  �ě�<49X<D��;�`B;�`B=C�;��
%   =���=�-<u<�h=�Q�=T��<���<T��=<j=C�<�j>	7L<���=49X<�1<��<�/<�j=e`B<�h<�h<�/=#�
=C�=�%=0 �=L��=+=,1=t�=#�
=}�=�\)=�o=T��>Kƨ=�o=��=�C�=��T=��=Y�=�E�=�E�=��-=���=�x�=���>%>��>�R>
=qB+�B	�BB��B"�B܇B	�"B��B	I�B|�B�XB#-B�tB�yB1�B�B!v|B�zB�B��B	fBC�B��B"�B ��B�`Bv�B)Bj!BOfB1�oB�B!��B%��B>B �B#��B(� BD_B�B�B�;B��B,�B�$B LB��BFSB�3B��B�A�kB�B?B"�OB��B�Bg0B�]BE�Bu�BL)B	��B�B�nB��B	��B�B	��Bt�B@B>�B�GB�FB�B�'B!N�B��B�*B��B	B�B�OB"�YB ��B�&B�B9/BΘB4�B1�vB�&B!��B%��B@kB ��B#ۭB(�#BB�BC{B?�B�*B��B,��B��B ?�B?dB@IB0YB�
B� A��B'B?�B"�PB��B?�B�DB@
B?�B��B
`IAɢ�C�y�A��cA�8�A���A�L�A�9A�?�A�9�A���A��+A�$A^-�A�\�A\�\A�\ A�~�A�(^A�3rA�K�AtJ'@�I\@� PA�~�A�
AaH�AO%iAreyAW��A89�@���@��@�Z�A29%A9-�@Z+uA��A��A��A��A��wAL�C��/A��B 9KA���@���B��B	�A�O�A���A9O�A�P�@\��@��C���?j�@H\@ǹCB
C�A��C�mAA��~A���A��A�{sA�)�A�{�A�e�A���A�A��Ab޲A�rA]^A�~�A��A�s+A�|�A��At�@�f@��A�W�A`CAaaAN,�Ar��AX>�A7�@�y@�$�@��A1:�A8W�@XA�A���A�p�A��A�A���A��C��A�BA�}�Aǁ'@��B��B	1�A�tA�)A9��A˄�@[��@�a1C���?P�}@ ��@��Z                  &   	      J   V         U   -   	      #      
   w                     %   
   	         
   '                        #         �            #         "   !         #      (   5   9   #            %               ?   ?         ;   +                  )                     %                                       %      3                                                                  !               !   '         )   )                                                                              %      -                                                      NU��O @+O���O�Z�OC�O�N/�gNI#�O�XePm=O��N��PO�r�O��N��N|h�N��3OS��N� UO���N3RuOcN'5�N$�tN�H�N[�$O��O��N���N�cO@ON��O3�N���O
FaN��N���Nc}�N��O���On��P(�aNDԥOE�7O�k�O���NIO*��O7�M��OXXOS�N�YhN�`O;N�/O;�O���O@�-OX�  �  �  �  �  �  >  �    �  f  �  �    Y    K  9  �  W  �  /  �  7  �  �  1  a  
  �  l  f  �    �  �  �  V  .  x  �    =  q  �  :  �  
�  6  �  �  �  �  $    g  `  �  �  	�  ���49X��C��49X�ě�<u�D���o=�P=C�;ě�<T��=,1<T��<T��<49X<ě�<�o<e`B=��<�o<�j<�t�<���<��
<��
<�h<��
<�1<�j<�`B<���=t�<�/=\)<�=o=o=\)=\)=�P=�w=<j=\=0 �=49X=P�`=T��=P�`=L��=�o=}�=�+=�C�=���=��
=�-=��=�`B=��`��������������������d_]_bgstu��������tgdlihjkinz���������zsl���������

����TUVYabnz}������znaUTYZ[dgt|�����}tgf\[YY��������������������Z[gtuyvtg^^[ZZZZZZZZ����������������������������� ����������)58BNSNMB@650)�������������������������)8?DB95)��)5BIIOPIGA5)����������������������������������������*))/<HJSOHF<7/****** ����&)*,+'" ��������������������GBBDIN[gty��~wtg[QG��������������������NOY[hqtz�����th[WON��������������������������������������������������������������������������������������$)+./)��56BOPW[\]][WOGBB?965�����

��������LJNOQ\hosohgb\SOLLLL����������������������������������������#0<CFIJIF@<0.# ]_aahhht��������th]]��������������������������

�������hno{{�������{nhhhhhh%$)5=BFJB;5)%%%%%%%%ABGLNX[egmoc_[VONHBA����������������������������
�����������
#105/#��������������������������������� 

������^]`amz����������zma^��������������������������������������������������������$)58:=85)&&%()+5565)&&&&&&&&&&3347;>HTW]^[TSH;3333������������� ����������������������������������������������������
���������#!
�������	

������������
������)6>ACB?;6)(,46<BFOS[_ge[OB63*(�0�=�?�E�=�<�0�&�$�"�$�)�0�0�0�0�0�0�0�0�n�zÇÓàæìíìèàÓÉÇ�z�n�j�b�i�nE�E�E�E�FFF$FFFE�E�E�E�E�E�E�E�E�E�����)�4�B�N�V�N�@�5�����������������������������������������������������������������������������������������������������������������������������������������n�p�q�p�n�b�U�R�M�U�b�k�n�n�n�n�n�n�n�n���	��(�0�4�2�"��	����������������������)�O�[�a�[�U�H�)������������������;�H�T�a�j�b�a�^�X�T�J�H�<�;�/�-�/�0�4�;�<�=�H�K�I�H�=�<�/�(�#�!�!�#�/�8�<�<�<�<�#�0�I�U�a�f�_�R�@�0�#�
��������������#�"�;�G�\�f�l�o�`�T�;�.�	��׾̾ɾھ�	�"���������������������������������	��� �"�&�(�"����	���	�	�	�	�	�	����������������������������������������������������������������������������5�A�N�X�Z�e�]�Z�N�A�@�5�4�2�5�5�5�5�5�5�6�B�O�[�h�p�w�v�m�h�[�O�B�6�)�%�� �)�6�6�8�>�B�D�B�6�)�!�%�)�4�6�6�6�6�6�6�6�6�����������ĿſĿ¿����������������������������ʼ˼˼ʼ�������������������������������������ֻܻܻ�������������������(�5�A�C�C�A�5�(���
���������ݿ�����������ݿܿܿݿݿݿݿݿݿݿݿ�"�.�G�T�^�f�d�^�T�G�;�.���������	�������׾̾ʾ������������������ʾ׾㿒��������������������������������������������	���	������׾Ծ׾ݾ����(�4�A�P�V�Q�E�A�4�(����
�����#�(�����������������������������{�������������������������������r�f�a�Y�Y�^�f�|��:�F�S�_�l�l�l�i�b�_�S�Q�F�D�:�6�1�1�:�:������$�(�*�(������������������(�4�A�M�U�W�M�A�4�(����������������� ������������������������t�o�t�w�v������(�+�5�6�5�(���	�����������5�A�N�g���������������s�A�(���&�+�-�5�ʼ�������	�����ּʼ��������������"�/�;�;�/�,�.� �	�������������������	�"�y���������������y�w�s�r�y�y�y�y�y�y�y�yD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�Ŕŭ����������ŻűŭŠŔōŇŇŋŊŊŐŔ�Q�T�T�R�W�O�@�6���
�����*�6�C�N�Q�zÆÇÇÇ�z�t�n�a�`�a�j�n�x�z�z�z�z�z�z�����������������������r�c�Y�U�Y�f�r�ƧƳ��������������������ƳƧƤƜƚƔƚƧ���$�%�$����	���������������������������������������������������������
���#�%�&� ��
����������������4�A�M�Q�R�M�M�A�@�4�(����(�)�4�4�4�4ÓàìñìäàÓÇÏÓÓÓÓÓÓÓÓÓÓ������!�0�9�-�&���������ۺ���S�_�l�w�x���������������x�l�h�_�S�H�N�SE�E�E�E�E�E�E�E�E�E�E�E|EuEiEhEaEiEuE~E������&�*�2�6�7�3�/�'�������޹ܹ���Y�e�r�~�����������������~�r�e�Y�U�Q�S�Y��'�4�7�@�K�M�V�M�F�@�4�'��������� D # : > &  L 8 8 ! 6 2 R \ F {   9 ;  g : K ; = 3 , ^ i , 8 ;  + + V c R � F 1 i ;  I ; l T B k 6 $ [ ` F w 2 B 2 C  v  Y  �  �  \    e  o  �  �  8  �  }  �  &  �  �  �  �  �  �  X  Y  I  �  n  E  ]  �  �  �  �  z     2  �  �  v    g  �  �  f  �  0  '  b  �  �    5  �  �    g  V  '  Q  �  �  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  �  �  �  �  �  �  y  n  c  U  F  8  &      �  �  �  �  y  <  h  �  �  �  �  �  �    h  I     �  �  H  �  �    �  �  �  �  o  V  H  6    �  �  f    �  �  ,  �  j  �  `  i  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  _  ;    �  V  m    �  �  �  �  �  �  t  a  K  0    �  �  �  N  �  p  H  �  �  �      *  1  6  =  <  *    �  �  �  S      �  �  �          �  �  �  �  �  �  �  �  �  x  e  R  >  *    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  P  �  �    K  w  �  �  �  �  �  �  �  �  V  �  7  :  J  Q  P  �  �    6  U  c  e  a  \  R  ;    �  �  "  �  �  C  u  �  �  �  �  �  �  �  �  �  �  �  }  h  Z  [  +  �  �  F   �  �    3  \  z  �  �  �  �  �  R    �  O  �  �  3  �  H  �  �  2  �    u  �  �            �  �    q  �  �  �      H  Y  N  3    �  �  {  `  V  0  �  �  n    �  K  �  O                    	  �  �  �  �  �  �  T  #  �  �  K  G  C  ?  ;  7  3  /  +  '         �   �   �   �   �   �   �  �  �  �      *  6  8  .    �  �  �  T    �  8  �  J  �  �  �  �  �  �  �  t  e  V  A  )    �  �  �  �  M    �  �  W  Q  K  B  9  -      �  �  �  �  �  �  \  +  �  �  X    	q  
8  
�  s  �  @  �  �  �  �  �  �  K    �  
�  	�  t  �  J  /  c  �  �  �  �  �  �  �  
    &  2  >  J  W  b  n  �  �  }  �  �  �  �  �  �  �  �  �  �  w  U  .  �  �  d  
  �    7  5  3  2  0  .  ,  %      	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  c  L  /    �  �  �  d  *  �  �  o  -  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  f  V    �  �  1  (          �  �  �  �  �  �  �  �  �  t  e  V  G  8  �    =  P  \  `  Z  M  9    �  �  �  �  N    �  f  �  V  
  
  
  �  �  �  �  �  �  j  N  2    �  �  �  �  v  M  #  �  �  �  �  �  �  �  �  �  �  �  w  f  Z  N  5    �  �  �  l  f  `  Z  T  O  L  I  F  C  =  5  -  %           �   �  Z  ^  a  c  e  d  _  U  B  +    �  �  �  s  Y  <    �  �  �  �  �  �  �  �  �  p  [  E  -    �  �  �  �  u  ?  �  �  �  �  �             �  �  �  �  �  �  i     �  5  A  4  �  �  �  �  �  h  H  %    �  �  �  x  o  D    �  �  �  z  �  �  �  �  �  �  �  �  �  �  �  �  b  >    �  �  �  L  S  �  �  �  �  �  �  �  �  �  �    s  f  W  E  3        �   �  %  A  Q  ?  ,      �  �  �  �  �  �  x  a  =    �  �  �  .  '  !                  !  )  2  :  
  �  �  B   �  x  g  W  F  6  &      �  �  �  �  �  �  �  �  �  |  f  P  �  �  |  q  a  P  >  (    �  �  �  �  v  l  )  �  �  p  @     �  �  �  �  �  �  a  3    �  �  Q    �  �  /  �  U    !  )  ;  5  /         �  �  �  �  �  U    �  |  �  l  �          !  9  R  \  e  m  p  p  m  c  P  .  �  �  `    �  p  �    E  j  �  �  f  )  �  R  �  '  B    �  �  �  �  :      �  �  �  �  �  y  W  2  �  �  �  F  	  �  �  �  �  �  �  �  �  �  �  �  �  v  \  :    �  �  u  (  �  d  �  d  �    �  	Q  	�  
\  
�    c  �  $  y  �  �  �  �  �  x  b  O       0  6  0  !    �  �  �  w  I    �  �  �  K    �  A  �  �  �  �  �  �  p  \  F  +    �  �  �  g  E  #    �  �  �  }  o  `  Q  B  4  "    �  �  �  �  �  �  |  c  J  1           d  �  �  |  s  d  R  4     �  ]  �  �  �  �  �  m  �  �  �  �  �  �  �  �  �  �  x  C    �  ]  �  m  �  h    �      �  �  �  �  i  I  (    �  �  �  n  K  )    �  �      �  �  �  �  �  �  �  �  u  e  T  B  1  !      �  �  I  `  d  V  C  *    �  �  h    �  L  �  C  �  �  �  
  "  `  \  Z  F  %    �  �  y  @  �  �  j    �  �  H  �  :  �  �  �  �  y  R     
�  
�  
k  
(  	�  	�  	1  �  g  �  �    �  '  �  �  �  �  m  K  6  "    �  �  n  !  �  E  �  �  �  �  V  	S  	�  	�  	�  	�  	�  	�  	�  	o  	6  �  �  H  �  j  �  _  �    5        �  �  �  S    �  �  R    �  A  �  6  �  %  �  s