CDF       
      obs    0   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�&�x���      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N$D1   max       P��Z      �  l   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �P�`   max       =ȴ9      �  ,   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?z�G�   max       @FG�z�     �  �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @vd          �  'l   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @O�           `  .�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�)�       max       @��          �  /L   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �@�   max       >�/      �  0   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B-      �  0�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�r�   max       B,��      �  1�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�m   max       C���      �  2L   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��r   max       C���      �  3   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max               �  3�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G      �  4�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7      �  5L   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�#   max       P`'�      �  6   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�'�/�   max       ?���!�.I      �  6�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �P�`   max       >7K�      �  7�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?z�G�   max       @E�\(�     �  8L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @vc33334     �  ?�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @M�           `  GL   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ͻ        max       @��           �  G�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         AP   max         AP      �  Hl   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Z�   max       ?���!�.I        I,         
   
      
         2   (         6         _         M   #      F   g   2   '         F        y      
         !      	   #                4         *   N�:yN�`sN���O0��N$D1N2�YOj�OwO��kP��wO7O�KP=>OJWO���P��TN6��NB{�P��ZO�	N4[ P׭P(��P?�O���N�ֿOd�O�wIP�_�Oy��O�UGN��SOjγO��eN@�LO�B�O�|�N���Oe��N5X�O�.O{� N.�PO˧�N��N�6�Oi~�N�,k�P�`�+<49X<49X<49X<49X<D��<T��<�o<�C�<�C�<���<�1<�9X<�j<�j<���<�/<�/<�`B<�h=o=+=C�=�P=�P=��=#�
=#�
=0 �=0 �=49X=8Q�=L��=]/=]/=m�h=u=�%=�o=��=��=�O�=�\)=��=���=�{=ȴ9�����������������������

���������$+/:;HOTUTTH>;//$$$$sqt���������������ts
#$($#
��������������������YY`mz����������zmgaY,./0<HIUanvneaUH<0/,WYaemz�����������zfW����)14)�������0055BJN[eghge[VNB950�������
��������������������������������������������������������������������/34<T\h����������O6/ )56<5+)          ���
adz�����76)���za��
#1<INOLG<0#
����������������������tuz���������������zt��������� �������������������������������5@EHFC962�A>=8BFOPW[]`c^[OKGDA�
!#&'))&#
������)25<>=5+���5B\��������gN5��������������������ns���������������znotuz�������zoooooooo�)5BMPKEB5)
�/0;HTYaglnkaXTLH>4//����������������������������'.1#���������#/<=:9-#���� #099200#        ���������������������� 

���������ttv|������������zttt��������

������� $()))(��������������������.*/0<HQTHE@<<;2/....=BDOQZ[htxvtmh[POB==������
!$.1-#
�����#*//1/%######�0�=�@�I�N�V�W�V�I�=�0�-�$�!�$�)�0�0�0�0�����������������������������������������"�/�1�;�<�?�;�1�/�%�"������"�"�"�"�f�s�z�~�����������s�f�Z�M�G�E�M�U�Z�fD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D������������������������������������������\�h�uƀƇƌƆƁ�u�m�h�\�T�O�H�C�<�C�W�\���������	������������������������tčēĚĝĥīĬĢĚčā�t�h�_�T�M�N�[�t�N���s�Z�W�N�5�����ݿ¿����Ŀ����(�N�b�n�{�{łŇŊŇ�{�s�n�i�b�W�U�S�R�U�Y�b�ûлջٻܻѻлû���������������������������6�C�J�I�>�*�����ŹŰŪűŹ������������� �'�/�4�7�4�.�'�����������5�A�Z�k�v�z�����������������N�A�-�)�-�5����M�f�������������r�Y�@�4����ڻѻ��s�����������v�s�q�s�s�s�s�s�s�s�s�s�s�����������������������������������������������"�:�_�l�g�T�������������������伋���������ɼ���������r�f�\�U�V�_�n������ ������ �����������àù����������������ìÞÎÎÔÚÖÛà�����ùϹ��������ܹù����������������(�4�7�.�,�/�(�����Ľ��������Ľ���"�;�G�R�\�_�]�T�G�;�.��	��۾�����"�����ʾ׾۾׾;ʾ������������������������Z�f�|���������������s�f�X�M�E�@�A�L�Z�#�0�H�N�M�I�D�<�0�(�
����������������#�O�hĆĳĲĚ�t�Q�B�6�/�#��
� ���)�6�O�G�T�`�m�q�y�z�s�m�`�T�G�;�:�3�0�2�8�;�G�����$�(�%� ���ֺɺ��������ֺ�����ŠŭŹſžŹŶŭťŠŜřŠŠŠŠŠŠŠŠ�N�T�[�_�f�]�P�B�5�)�����$�,�5�B�I�N�@�I�^�b�S�I�?�0�#��
�����������
�#�0�@�
����#�(�#��
��������
�
�
�
�
�
�
�y�����������������������y�l�e�]�[�`�l�y�t�t�k�g�j�g�[�N�B�;�;�V�[�i�t���������ĺź�������������������������������'�,�'�"�������ܻۻ׻ջٻ������"�%�"��	�������	������������"�/�3�6�1�/�(�"���	���� �	����������������	�����������������Y�e�r�r�v�r�e�Y�N�L�E�L�O�X�Y�Y�Y�Y�Y�Y���ʼּ��߼ּʼ���������n�\�f�������¦²¾¿����¿²¨¦¦¦¦¦�'�/�3�;�@�L�M�U�V�L�@�<�3�0�/�'�&�&�'�'EuEvE�E�E�E�E�E�E�EuEiE\E\EbEbEdEdEcEiEu�O�C�6�6�*�(�����*�6�C�O�X�O�O�O�O�O K L F / R 5 3 V  c ) > R ' f W 7 P j  c ) A C 4 w a L L $ 0 ; 6 N M ` f 3 W T & O Y ? M ~ V >  �  �  �  x  ]  K  �  w    �  >  V  �  �  O  l  U  o  �  o  `  e  �  �    7  n  e    �  @  �  �  c  p  Q  S  �  8  T  =    m  �  �    2  ʽ@���<��
<�1<�t�<�1<�j=t�=��=e`B=+=��=���=�w=L��=��<�=o=��=�%<��=��>
=q=�1=���=49X=H�9=�G�>�/=�hs>&�y=H�9=]/=���=m�h=�-=�t�=�O�=Ƨ�=�C�=��-=Ƨ�=��-=���=�j=�j>J=�G�BD�B��A���B$�BIB"!�A���B7B NjB?�B'�B#H�B��B!��BB��B��B<MB�B%u�B�B�IB�>B"r�B�dB�}B$��B�(B&3B"�B��B B��A�&�B�B-B�	B%t#Bl�B��B
��B�B��BZ�BƪBk.Bj{B��B?�BSrA�r�B
ӘB��B"=�A��B54B F�B.B�B#A�BM�B!��B �B?�B�tB>BʌB%��BUwB(�B�[B"��BOBYB$��B@nB�B;�B��B >�B&nA�~�B��B,��B6@B%�xB��B��B
AB<�B��B>B��B�/B�EB?5B
�A�9�A���ABIIC��@:kB0�Aҙ�A�&�A�s�A�\�@���A��@���A�=H@ӌEAEF�A�'OA���@�p�A�~�Aκ�>�mA,$A`hAL�7A@��A���A�d�Af��@L��A�SA�&oA�3�A��qAi�A�b�@!ÿ@���A[�FA��A�ԭ?��@�*�A��?���C���B MBB
QA��:A���AB�:C��	@�vB<
AҀ�A�}�A��A��^@�^A�|t@��`A��@�}#AE�	A�U�A��N@�'�A��/AΉE>��rA+ �Aa �AI:�A>��A���Aڂ�Af�G@K�'A�}gA�;6A��A�Q$A�[A�j�@$A@�%A[)�A��7A�x�?� |@�܍A��c?��C���B ��         
                  2   (         7         `         M   $      F   g   3   '         G        y      
         "      
   #         !   	   4         +                                 A         '      '   ;         G   !      %   )   -   #         %   ?      %               #   !                     %                                          3                  7         1   !         !                                          !                                 N�:yN�`sN���N��.N$D1N2�YOj�Nv-Op�P>)�O7O00O�ZOJWO�%P`'�N6��NB{�P;ڃO���N4[ O�� Oʔ�O��uO���N�ֿN�tvO�U�O���OkrtOjN=N��SOjγO/BRN@�LO	�O�|�N���O\�+N5X�O�.O8��N.�PO��N�#N|f|O_N�,k    �  �  /  
  a  _  {  W  B    �  9  �    �  0  F  )  �    	P  \  h  D  �  O  	�  _     5  �  �  �  �  �  �    P  P  �  �  �  <  �  �  (  νP�`�+<49X<u<49X<49X<D��<�1<�/<���<�C�<��
=��<�9X<�/=+<���<�/=<j<�<�h=aG�=}�=H�9=49X=�P=�w=u>7K�=49X=�9X=49X=8Q�=ix�=]/=�O�=m�h=u=�o=�o=��=�hs=�O�=���=���=��w=��=ȴ9�����������������������

���������$+/:;HOTUTTH>;//$$$$��������������������
#$($#
��������������������YY`mz����������zmgaY805<HLUZURH<88888888jfgmnvz����������zrj���������!�������0055BJN[eghge[VNB950��������
	�������������������������������������������������������������������58@Sht���������t[B75 )56<5+)          ���
������").)�������
#0<IKMNKF<0#
 ������������������������������������������������������������������������������������)5<@@>;6)��A>=8BFOPW[]`c^[OKGDA� 
#'(&##
��������)057854)��%#&,5BN[douupg[NB5+%����������������������������������������otuz�������zoooooooo�)5BMPKEB5)
�97;;HTZachifa]THB=;9����������������������������
������������#/<=:9-#���� #099200#        ���������������������� 

���������ttv|������������zttt��������� ��������� $()))(��������������������-/6<HIOH=<:/--------JOR[\htwutkh[OJJJJJJ������

������#*//1/%######�0�=�@�I�N�V�W�V�I�=�0�-�$�!�$�)�0�0�0�0�����������������������������������������"�/�1�;�<�?�;�1�/�%�"������"�"�"�"�f�s�~�����s�f�Z�U�Z�]�f�f�f�f�f�f�f�fD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D������������������������������������������\�h�uƀƇƌƆƁ�u�m�h�\�T�O�H�C�<�C�W�\�������������������������������������h�tāċčĚĜģĦěĚčā�t�h�^�U�U�[�h�A�N�a�j�d�Y�R�A�(���ݿֿϿ߿����(�A�b�n�{�{łŇŊŇ�{�s�n�i�b�W�U�S�R�U�Y�b�����ûл׻ڻллû����������������������������)�8�8�+��������������Ź�����߻����� �'�/�4�7�4�.�'�����������5�A�N�Z�d�o�r�}���������Z�N�G�A�0�,�/�5�'�M�f������������r�Y�@�4�����ڻ�'�s�����������v�s�q�s�s�s�s�s�s�s�s�s�s��������������������������������������������"�J�\�_�]�T�;�������������������������������������������r�f�]�V�X�b�q������ ������ �����������ìù����������������������ùìáÝáäì���Ϲܹ��������Ϲù���������������������������ݽĽ����������Ľݽ�.�;�J�U�Z�X�R�G�;�.�"��	� �������"�.�����ʾ׾۾׾;ʾ������������������������M�Z�f�s�z�������s�f�Z�Y�M�F�A�M�M�M�M��#�0�4�<�<�8�/�#��
���������������
��O�[�h�t�ąă�|�t�h�[�O�B�:�5�3�4�:�B�O�T�`�m�p�x�y�r�m�c�`�T�G�;�4�0�3�9�;�G�T��������������ֺܺɺƺɺպ���ŠŭŹſžŹŶŭťŠŜřŠŠŠŠŠŠŠŠ�N�T�[�_�f�]�P�B�5�)�����$�,�5�B�I�N�#�0�<�=�H�C�<�5�0�#��
��������
���#�
����#�(�#��
��������
�
�
�
�
�
�
���������������������y�u�m�l�i�j�l�y�����t�t�k�g�j�g�[�N�B�;�;�V�[�i�t���������ĺź�������������������������������'�'�!���������ܻػջٻ������"�%�"��	�������	������������"�/�3�6�1�/�(�"���	���� �	������	��������	���������������������Y�e�r�r�v�r�e�Y�N�L�E�L�O�X�Y�Y�Y�Y�Y�Y�������ʼ׼ܼݼּʼ���������r�f�r������²¸¿��¿¿²¬¦£¦¬²²²²²²²²�3�7�@�L�L�T�T�L�@�>�3�1�1�(�3�3�3�3�3�3E�E�E�E�E�E�E�E�E�E�EuEiEgEfEiEjEnEuEyE��O�C�6�6�*�(�����*�6�C�O�X�O�O�O�O�O K L F : R 5 3 =   l ) ? V ' X ] 7 P N  c  7 < & w O 7 ! !  ; 6 ; M J f 3 T T & A Y : E m 8 >  �  �  �  �  ]  K  �  |  �  �  >  /  �  �  �  	  U  o  �  Q  `  '  �  I  C  7    G  <  �  �  �  �  t  p  ^  S  �  �  T  =  �  m  h  E  �  H  �  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP  AP    �  �  �  �  �  �  �  �  �  x  j  [  M  >  ,       �   �  �  �  �  �  �  �  �  �  �  �    z  v  k  R  :  !  	   �   �  �  �  �  }  t  e  T  ?  )       �  �  �  p  0  �  �  G   �    	          $  +  -  )         �  �  �  �  �  �  �  
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  a  T  G  ;  /  $         �  �  �  �  �  �  �    F  �  �  _  T  J  A  9  1  )  !              �  �  �  �  �  �  �  �  �  %  U  j  x  y  n  X  4    �  �  G  �  �  U  �  �  �  �  3  N  W  Q  C  )  �  �  y  (  �  \  �  m  �    �  �  �  �  �  "  :  @  +    �  �    �  �  �  �  b    �  �   �        �  �  �  �  q  L  $  �  �  �  n  6    �  �  �  �  �  �  �  �  u  `  B  !  �  �  �  �  �  �  F  �  ~    �    B  �  �  �     4  9  /    �  �  �  `  �  e  �  T  �  }    �  �  �  �  �  �  ~  k  Y  H  9  2  5  8  8  7  8  :  =  @  �        �  �  �  �  �  �  �  �  u  ]  P  ;  �  �  G  �  k  �  �  �  �  �  E  /  8    �  ,  �  �  6    �  K  l    0  2  4  6  9  :  7  5  3  1  .  +  (  $  !            F  9  +      �  �  �  �  q  O  -  /  I  c  }  �  �  �  �  �  �  �    �  )    �  �  �  �  P     �  n    �  �  �   �  �  �  �  �  �  �  y  [  G  ?  6  +      �  �  �  X  �            �  �  �  �  �  �  �  �  �  �  �  }  s  j  a  X  O  �  �  	  	  	-  	=  	N  	L  	7  	  �  �  �  H  �  S  q  Y    �  	�  
1  
�  
�  9  Y  Y  D    
�  
l  	�  	l  �  h  �  �  �  �  Q  2  �  �    D  `  h  R  /     �  �  �  p  F  �  ~  �  �   �  �  5  =  C  C  ?  6  '    �  �  �  Y    �  r    �  %  {  �  �  �  �  �  �  �  �  �  �  �  �  �  q  X  @  (  9  [  |  �  -  M  F  9  (    �  �  �  �  �  e  9  
  �  �  }  k  a  	-  	j  	}  	�  	�  	�  	�  	�  	z  	Q  	  �  �  1  �  T  �  �  �    n  S  u  V  Y  �  �  q  �  ?  _    v  �  s  �  �  	  ,              �  �  �  �  �  r  O  '  �  �  �  �  P    �  9  N  �  ]  �  �    4  '  �  �  a    �  �  
�  	�  U  �  �  �  �  �  �  v  e  S  B  7  6  5  5  ,         �  �  �  �  �  �  �  �  �  �  �  x  `  S  P  L  E  =  2  $  	  �  �  �  �  �  �  �  �  �  �  �  �  �  c  0  �  �  W  �  �  ;  �    �  �  �  �  �  �  �  �  �  �  �  �  �  }  e  C    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  *  �  �    �  }  l  T  9    �  �  �  d  *  �  �  �  ]  &    �  ~                   �  �  �  �  �  �  ~  �  �  �  �  �  �  M  K  4    �  �  �  _    �  �  7  �  �  $  �  "  �  �  �  P  ?  /      �  �  �  �  �  �  �  q  ]  H  '     �   �   �  �  �  �  �  �  t  X  9    �  �  �  �  �  �  o  K    �  �  �  �  �  �  �  �  �  �  o  H    �  �  V  �  �  %  �  �     �  �  �  �  �  �  �  �  j  P  7      �  �  �  �  o  O  .    )  9  6  +      �  �  �  w  +  �  m  
  �  �    .  }  w  J  E  4  ]  �  �  �  �  �  c  3  �  �    �  9  �  ]  �  �  �  �  t  G  3  -  �  �  w  .  �  q  
  �  R    �  �  V  �      #  '  $    �  �  �  k  5  �  �  �  +  c  �  �  �  �  �  �    j  T  8      �  �  �  �  s  L     �  �  �  Y