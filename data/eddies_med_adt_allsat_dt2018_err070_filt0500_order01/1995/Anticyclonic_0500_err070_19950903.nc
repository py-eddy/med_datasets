CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�\(��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�j   max       Pǁ�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��h   max       =      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?.z�G�   max       @FL�����     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min                  max       @vu\(�     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @R@           p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�2        max       @�A`          �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��/   max       >���      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B0�      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�{�   max       B1/�      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�d   max       C�hh      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C�g�      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         U      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          C      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          ?      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�j   max       P�%�      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�䎊q�j   max       ?�"h	ԕ      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��h   max       >+      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?.z�G�   max       @FL�����     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vu\(�     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @          max       @R@           p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�2        max       @�:�          �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A)   max         A)      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�L�_��   max       ?�"h	ԕ     �  Pl      
     T   
      #   .         $   ?      [      #            -       #            0      /      Q            	   )               
   '   ;   	            T      Y   -   	   	            M�jO3��O($�P�zN�M�Oy�PO�bO��NON��qP<�P`cOP�P�2�NAR�O�`�O/��O)/�NB^�PD��O��rO��qN+�N#4�N�c�PB��O��P$Q�N
ePǁ�N�F�P%rDO���N3�PL��O�?O^�N�ĥNC�Op1O�X'P�CM��6NA�@N�i?N���OTW�OBXQO�&vO�kN0��N��N��N �4M���NsH���h��C���o�#�
�t����
%   :�o:�o;��
;ě�;�`B;�`B<o<#�
<�C�<�C�<�t�<���<��
<��
<��
<�1<�9X<�9X<�j<ě�<���<���<�/<�/<�h<�=C�=t�=��=�w=,1=P�`=P�`=e`B=ix�=��=�+=�+=�O�=�hs=�t�=�t�=���=��T=�1=� �=�v�=���=����

�����������	!,25:A?<5*)�������������������� +5Na��������gNB)��������������������ZWX[gt����������tg`Z����zcURPUanz������������zaYVWanz�������

#*0280/&#


 #/0()6O[lrrto[SB) |z�����������������|��������������������������)5Y\P5)����)05;5)�����)6AB8-���$*.356COR\alhe]OC:*$,+-/4<HQUdnlaUMH<4/,��������������������LFH[t�����������j[RL�����#/<;@ID/#
����������
#')'#
 �����������������������BBOU[fhkiha[SOLBBBBB��������������������ljlz�������������zql����������������������������� 

 ������ #)08720#wt����,49/�����}w20046;BHOPRQPOBA8622��������� ����������������
"#$
���stt��������tssssssss)B[eih[QB)�������� ����������  #02<=IQTSII<0#  (%&)/<@FF@</((((((((������������������������������������������������� ���������������)42,,$!���")6BLEB65)""""""""""����� ���������������	))*)$���%)-.0/-)%����������		������/./6;?CHTachge`[H;9/������
#'153/#
���rnosz�������������zr��������������������	
#'.-#
						aaW^anurnaaaaaaaaaaa[anz~�znfa[[[[[[[[[[		

								srst��������tssssss�4�@�M�O�M�L�@�4�3�0�4�4�4�4�4�4�4�4�4�4���ʾ׾������׾Ѿʾ������������������'�3�@�L�T�Y�f�g�e�Y�L�@�9�3�'�%���!�'�)�hčĦİİĦĕč�t�[�6�"��	������)�������Ⱥź���������������������������������������������������������������������E�E�E�E�E�E�E�E�E�E�FFFE�E�E�E�E�E�E����������������������	��������������ļ��������������������������������������������������������������x�r�l�c�l�x�������A�M�Z�s����������ʾ���f�A�$�4�7�2�A�����
�)�;�D�:�)������äèöú����ô�Ż���-�;�:�5�-�!�����������������#�<�U�b�y�z�u�w�u�b�U�
�������������
�#�����������������������������������������	�"�;�G�^�d�e�a�e�[�T�G�;�"�	� ������	�;�N�T�`�m�p�v�q�m�c�`�T�O�G�;�2�-�.�3�;������
�������������������������������������������������`�y�����������������y�T�;���"�.�;�T�`���9�A�F�L�S�X�U�A�(������������m�y�����������������y�m�`�T�N�M�T�W�`�m�.�;�D�G�J�G�;�.�#�'�.�.�.�.�.�.�.�.�.�.�H�H�M�U�[�a�f�a�U�R�H�A�<�9�<�H�H�H�H�HÓçìóùüùõìàÓÇÁ�ÇÔÚÓÑÓ���������������������s�g�N�C�:�C�H�Q�g���5�A�N�Z�g�m�s�k�g�^�Z�N�C�A�5�-�(�(�2�5�������*�2�F�L�O�C������űũŭŷ���߼������ ������ݼ�����������������"�=�G�B�/�"�	�������������������������!�+�!���
������޺޺����������(�9�G�G�A�1������ݽ������������[�t¦¨¦�t�g�[�N�A�5�3�E�[�T�`�`�m�m�m�i�`�T�Q�N�R�T�T�T�T�T�T�T�TƁƚƧ������=�G�E�8�&��������ƚ�~�{Ɓ�H�U�V�Z�Y�U�O�H�<�/�#�����#�-�/�<�H�ֺ���������������ֺֺκκӺֺ־Z�f�s�������s�f�Z�Q�O�Z�Z�Z�Z�Z�Z�Z�Z�����
���
�����������������������������S�V�`�l�y�����������v�l�`�[�U�S�S�U�R�S�ùϹܹ������������ֹ����������û��û����� ����ܻû��x�l�_�T�P�_�x���'�*�*�2�'�����$�'�'�'�'�'�'�'�'�'�'¿��������������¿¾¾½¿¿¿¿¿¿¿¿�L�Y�e�k�r�y�~�����~�~�r�n�e�[�T�N�L�K�L�N�[�g�t¢�t�g�[�R�N�I�N�ND�D�D�D�D�D�D�D�D�D�D�D�D�D{DqDqDtD{D|D�ĦĳĶĿ��������������������ĿĳĚĘĚĦEuE�E�E�E�E�E�E�E�E�E�E�EuEnEkEjEjEkEkEu�ʼּ���������� �����ּʼ���������E7ECEPETEYEPECE:E7E4E7E7E7E7E7E7E7E7E7E7ǡǩǭǯǭǩǡǔǈǂǆǈǔǟǡǡǡǡǡǡ������%���
����������������������������������������������������O�\�h�u�v�u�h�\�Z�O�C�M�O�O�O�O�O�O�O�O�'�4�@�F�M�M�M�B�@�4�*�'��%�'�'�'�'�'�' 9 H . $ 0 F 9 E a W Y & ; % m 5 7 8 9 6 6 + > � L 4  A B @ V T P O ^ I . 3 a ; X P � L U i ( l 8 @ Y I T i � *    �  p  �  �    _  �  P  �  {  �  �  .  �  �  �  o  ]  w  8  �  O  �  �  Q  Q  �  *  �  �    B  g  �  X  A  �  a  [  _    �  u  <  X  �  �  U  �  g  �  4  q  E  w��/�o;�`B>���:�o<#�
=\)=@�<o<D��='�=�O�<�9X=ȴ9<T��=T��<�9X=<j<�j=�o=P�`=aG�<ě�<���=\)=�hs=#�
=�t�<�`B=�"�=�w=q��=e`B=,1=��-=H�9=�o=H�9=aG�=y�#=\=�=��=�\)=��w=���>�-=�Q�>#�
=�=�E�=�v�=�v�=ě�=�"�>I�B#��B�nB!^(B��B"T�B
1B-cBd�B1B% �B�DB�zB ��B IB}�B�!B0�B0BJ�B
eBCB�KBwB��B!��B �fB�B�B%��B��B5�B!�4B�<B�WB%�B&_B&�B|�B��B,��B�B��B��B��B8iBSB|�A��B�B�cBH�Bo,BxB�XB�zB�2B$;$BA�B!Q~B�rB"@�B	� B 9BA�B?�B$��B�&B�}B F[B�B΅B�B1/�BגBL{B
],B��B:�B�uBF�B":�B �?B��BB�B%yXB�>B6CB!��B�B�"B��B/B&=TB��B%�B,�B�B��B4jB��BF�B<�B��A�{�B��B>B��B�OB�>B�>B�=B�@Ѹ�AP�?�~�A�&@$�~A��lC�hhA��h@�r�@�QAJ�fAҌ�@b*|AꝀAs�uAaH�Ae�=A҆_A�;tAhwcA�eAl�Ab6�A�}[A�<�A��A�e�A��rA��A��@X��A.;�A�&Ag��B�AÁt@H��AA��A���A@>�d@�5?�~A�?虇A�{EC���A���C��Av C���BqmA3�z@ ��B�@Ε@���AR�*?�6,A�z�@$A��C�g�A��@�	�@��sAK�AҐd@bn�A�9Ar��A`��Ae
�A�}&A�~Af�*A�">Ak�AbܤA�]�A˄�A���A���A�A+A��R@[��A.G,A�Ah��Bq-A�&�@J�ABJ�A��FA�>��@���?��A��V?�03A��PC��JA��C�KAӋC���B?@A4�D@soB�@��      
     U         $   /         $   ?      [      $            -       $            1      0      Q            	   *                  (   ;   	            T      Y   -   	   
                        9            !         1   3      1      !            -   '               +      +      C      /   !      1                     +                                 
                                             )      %                  !                  !      #      ?      /   !      +                     '                                 
         M�jNب:N��_O���Nd :O3��O�bO<w�NON��qO��P#��OP�P	��NAR�O-ΪO/��N�;�NB^�O�6{O��@O ��N+�N#4�N�c�O�,�N���O�v�N
eP�%�N��P%rDO���N3�P ��N���O^�N�ĥNC�Op1OB�sO��1M��6NA�@N�i?N���O?k�OBXQO]��Ol��N0��N��N��N �4M���NsH�  =  �  �    D  �  1  5  4  �  �  l  �  �      q  1    ;  �  �  S  �  j  �  �  �  f  �  �    8  �  �  O  �  O  �    6     i  �  |  �  �  F  �  �  �  f  "  K  �  ��h�e`B�o>+�o�o%   <�C�:�o;��
<�9X<���;�`B=��<#�
<��<�C�<���<���<��<�/<�h<�1<�9X<�9X=\)<�h=C�<���<��<�h<�h<�=C�=,1=�w=�w=,1=P�`=P�`=}�=�C�=��=�+=�+=�O�=��-=�t�=�9X=�{=��T=�1=� �=�v�=���=����

�����������&).35985)"��������������������52127BN[gt|��~tg[NB5��������������������Z[^gt���������tgd\[Z����zcURPUanz�������a^_afnz���������znma

#*0280/&#


5016ABO[adfhhf_[OB95����������������������������������������������5AFHB5���)05;5)���� ����$*.356COR\alhe]OC:*$//2<HUX`USH=<;0/////��������������������XSQXgt����������tg^X�����#/79<?8/#
��������
#$&##

�����������������������BBOU[fhkiha[SOLBBBBB��������������������xsqu{�������������zx������������������������������������� #)08720#wx����)27-�����w3126BENOQONB<6333333��������� ����������������
"#$
���stt��������tssssssss")5Xafd[NB5)�������������������  #02<=IQTSII<0#  (%&)/<@FF@</((((((((���������������������������������������������������������������� #+,'('#���")6BLEB65)""""""""""����� ���������������	))*)$���%)-.0/-)%����������������/./6;?CHTachge`[H;9/�����
"+/0/-#
���zwz����������������z��������������������	
#'.-#
						aaW^anurnaaaaaaaaaaa[anz~�znfa[[[[[[[[[[		

								srst��������tssssss�4�@�M�O�M�L�@�4�3�0�4�4�4�4�4�4�4�4�4�4�ʾ׾ھ����׾ʾƾ��������������Ⱦʾʺ'�3�?�@�L�W�P�L�@�3�'�%�"�&�'�'�'�'�'�'�B�O�[�h�tĀąĄ��t�h�[�O�B�3�-�*�+�1�B�������ƺº���������������������������������������������������������������������E�E�E�E�E�E�E�E�E�E�FFFE�E�E�E�E�E�E��������������������������������������Ѽ��������������������������������������������������������������x�r�l�c�l�x�������������˾׾ؾپѾʾ����������s�k�i�x�����������(�3�<�1�&��������ùðõ�����޻���-�;�:�5�-�!�����������������#�0�<�I�U�Y�\�[�T�I�<�#������������
�#�����������������������������������������.�;�G�K�T�T�R�J�G�;�.�'�"���
���"�.�;�N�T�`�m�p�v�q�m�c�`�T�O�G�;�2�-�.�3�;����������������������������������������������������������`�m�y�������������y�m�T�G�(�%�%�0�;�T�`����5�@�D�G�D�5�(��������������m�y���������������}�y�m�`�`�T�T�T�`�a�m�.�;�D�G�J�G�;�.�#�'�.�.�.�.�.�.�.�.�.�.�H�H�M�U�[�a�f�a�U�R�H�A�<�9�<�H�H�H�H�HÓçìóùüùõìàÓÇÁ�ÇÔÚÓÑÓ�g���������������������������Z�R�R�W�a�g�N�Z�\�g�g�g�b�Z�Q�N�L�A�5�3�0�5�A�C�N�N���������*�<�B�A�6��������ŹűŴ���߼������ ������ݼ�����������������"�<�E�@�/�"�	�����������������������������������������������������(�9�G�G�A�1������ݽ������������[�t¦¨¦�t�g�[�N�A�5�3�E�[�T�`�`�m�m�m�i�`�T�Q�N�R�T�T�T�T�T�T�T�TƧ���������0�<�9�$��������ƣƓƈƈƎƧ�H�T�U�Y�X�U�N�H�<�/�#���#�/�/�<�B�H�H�ֺ���������������ֺֺκκӺֺ־Z�f�s�������s�f�Z�Q�O�Z�Z�Z�Z�Z�Z�Z�Z�����
���
�����������������������������S�V�`�l�y�����������v�l�`�[�U�S�S�U�R�S���ùϹܹ������������ܹϹù������ûܻ��������ܻлû������x�r�_�[�l���ú'�*�*�2�'�����$�'�'�'�'�'�'�'�'�'�'¿��������������¿¾¾½¿¿¿¿¿¿¿¿�L�Y�e�k�r�y�~�����~�~�r�n�e�[�T�N�L�K�L�N�[�g�t¢�t�g�[�R�N�I�N�ND�D�D�D�D�D�D�D�D�D�D�D�D�D{DsDsDwD{DD�ĦĳĶĿ��������������������ĿĳĚĘĚĦE�E�E�E�E�E�E�E�E�E�E�E�E|EuEpEnEnEpEuE��ʼּ�������������ּʼ�����������E7ECEPETEYEPECE:E7E4E7E7E7E7E7E7E7E7E7E7ǡǩǭǯǭǩǡǔǈǂǆǈǔǟǡǡǡǡǡǡ������%���
����������������������������������������������������O�\�h�u�v�u�h�\�Z�O�C�M�O�O�O�O�O�O�O�O�'�4�@�F�M�M�M�B�@�4�*�'��%�'�'�'�'�'�' 9 Q +  0 D 9 ( a W D  ;  m " 7 * 9 ( ) 0 > � L /  C B > J T P O _ F . 3 a ; T W � L U i ' l 4 E Y I T i � *    �  �  f  x  �  _  �  P  �    �  �  [  �  k  �  �  ]    w  U  O  �  �  ;  �  G  *  o  �    B  g  %  (  A  �  a  [  �  c  �  u  <  X  �  �  �    g  �  4  q  E  w  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  A)  =  ;  :  8  6  4  3  1  /  -  ?  d  �  �  �  �    D  i  �  �  �  �  �  �  �  �  �  �  y  e  M  3    �  �  �  �  �  |  _  }  �  �  �  �  �  �  �  �  �  �  j  <  �  �  B  �     �  G  `  �  �  c  �  M  [  $  �    �  V  �  j  �  �  �  F  o  >  B  D  D  B  =  6  -  !      �  �  �  �  a  :    �  �  �  �  �  �  �  �  �  �  �  �  r  ]  C  &    �  �  �  z    1    	  �  �  �  �  Y  %  �  �  a    �  ]  �  w     f  �  g  |  �  �    )  1  1    �  �  �  ;  �  T  �  1  q  d   �  4  ,  %      �  �  �  �  �  u  ]  D  +    �  �  �  �  �  �  �  ~  }  |  z  ~  �  �  �  �  �  �  �  v  g  X    �  c  3  Z    �  �  �  �  �  �  �  �  �  }  O    �  �  7  �   �  �    P  g  k  ^  B    �  �  K    �  s    �  3  �  �  "  �  �  �  |  v  e  _  [  X  R  H  A  8  1  +  !  �  �  h    �  6  t  �  �  �  �  �  �  �  z  ]  6    �  F  �  �  �      	  �  �  �  �  �  �  �  �  �  �  �  ~  n  ^  N  >  .    �  �  �  �  �            
  �  �  �  �  �  >  �  K  ~  q  i  b  Z  R  J  B  :  4  0  ,  )  &  $  !    	   �   �   �  F  }  �  �  )  0  )      �  �  �  |  M    �  �  4  �        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      .  7  ;  6  2  *    �  �  {  -  �  L  �  �  O  ~  }  �  �  �  �  �  �  �  �  h  C    �  �  v  :  �  �  N  �    F  �  �  �  �  �  �  �  �  ^  /  �  �  [  �  �    �  �  S  O  J  E  @  <  7  0  (             	           &  �  {  u  n  h  a  Y  Q  I  A  =  =  =  =  =  6  -  $      j  g  _  S  ?  -    �  �  �  �  �  �  �  �  �  �  w  s  w  e  �  �  �  �  �  �  �  �  �  _  1  �  �  k     �    6  p  �  �  �  �  �  �  �  �  �  �  �  �  ~  d  C  !  �  �  Y  �  _  �  �  �  �  �  �  y  G    �  �  O    �  L  �  7  �  �  f  j  m  p  s  v  y  z  z  y  y  y  x  t  j  _  U  J  ?  5  i  �  �  �  x  c  W  1  !  �  �  ^    �  4  �  �  �  �  c  �  �  �  �  �  �  �  �  �  �  �  �  z  j  Z  Q  L  O  S  X    �  �  �  �  �  d  K  /  
  �  �  �  �  t  e  `  >       8  '      �  �  �  �  �  �  }  A    �  \  �  �  �  $   h  �  �  �  �  �  �  �  �  �  �  �  �  �  u  _  H  0    �  �  v  �  �  �  �  s  ]  F  +    �  �  �  h    �  �  ;  w   �  K  M  M  H  >  .    
  �  �  �  �  |  V  -  �  �  �    f  �  �  �  �  �  t  ]  E  $    �  �  �  �  A  �  �  t  D  �  O  J  E  ?  9  3  ,  $         �  �  �  �  �  �  �  �  �  �  �  q  a  P  @  .    	  �  �  �  �  �  �  l  =    �  �    �  �  �  �  u  Y  =  ,  +  .  5  /         �  �  r   �    )  -  5  1    �  �  �  k  2  �  �  �  A  �  <  R  T  A  �  �  �  �  �  �  �  �    W  )  �  �  X    �  s  �  g  c  i  S  =  -         �  �  �  �  �  �  X  -  �  �  �  R    �  �  �  �  �  �  �  �  �  �  �  w  [  ?  #    �  �  �  �  |  n  _  Q  A  0      �  �  �  �  j  P  M  :    �  �  S  �  �  �  �  �  �  j  O  ;  '  	  �  �  p  )  �  �  L  �  �  �  �  �  �  �  �  �  Z     �  �    R  �  �  �  
�  	�  �  �  F    �  �  �  X  +    �  �  �  t  Q  #  �  �  p    �    �  U  �  �  �  �  q  L    �  f  �  L  �    
c  	�  o  �  ^  �  �  �  �  �  �  �  �  �  �  �  �  V    �  A  �  >    �  �  �  �  ~  j  X  J  <  0  $      �  �  �  �  �    '  F  f  R  =  $  	  �  �  �  �  f  =    �  �  �  U    �  Z   �  "    �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  Z  <    K  E  ?  9  3  -  '  "          	       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      �  �  �  �  �  p  >    �  p  "  �  q  	  �  +  �  A