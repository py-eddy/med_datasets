CDF       
      obs    A   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��t�j~�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N �   max       P��
       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��{   max       =t�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>O\(�   max       @F������     
(   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�`    max       @vW�z�H     
(  *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @O�           �  5   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ͱ        max       @���           5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �C�   max       =C�       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B4�@       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�x�   max       B4��       8�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >���   max       C��}       9�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�h   max       C���       :�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          a       ;�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          A       <�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          ;       =�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N �   max       P���       >�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����+   max       ?���E��       ?�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��{   max       =t�       @�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>O\(�   max       @F������     
(  A�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vW��Q�     
(  K�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @L            �  V   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ͱ        max       @�'�           V�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D+   max         D+       W�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ᰉ�(   max       ?��e��ں     P  X�               2      
      %                  8      A   .      	         `   8               	            	      
               .      J   7      "   .            (                
      L      9   I   
      
      	N �O^��O?�POL�?PX�N��O7SJN�<O���Of%N��8OY�zO�Y�N�c�P��*N>y�P�"0PqV�O��N��O9�OV� P��
P-6NQ�KO�Z�O��O���N�@�N%9N��OA�O%�O��NC�>ObN�N�'O�{�P��N@^IO�8�Px1O!��P��P*m�N��GN�}�N=�rOs9GN�dN�|�O�ʄOn��OG�O���P9X�Nn��PPC�>N�~:O^�1Ns��N�b�N*z=t�<t�;ě�:�o:�o%   ��o�o���
�ě��o�o�49X�49X�T���e`B��t���t����㼴9X��9X��9X��9X���ͼ�������������`B��h��h�����+�C��C��t��t���P��P��P�,1�0 Ž49X�49X�8Q�8Q�8Q�D���L�ͽT���Y��]/�e`B�q���q���u�u�}󶽁%��o��hs���罩�罩�置{��������������������#388C=#!
 
@BN[^gkomig_[NEA76:@���������������)5FN[g{�y[NB5� $"������������	
	������������������������������������������������S[ht��������tpohec[S"$/1:;=;/""(/6KO\gh_[UOMC6*%"$([`ht����������t[UQS[��������������������/;H\g�������zaLJ7($/Z[^hhmtxth[VTZZZZZZZ����#Sn}���zgI0���������������������
#/:<<<7/-# 
xz���������~zyxxxxxx����������������������),0-)��������#0Un��{bI>4*����+5B[~�������tNB5)#$+yz|�������}zyyyyyyyyz�����������������zz���������������}z}��&5Ngt������tgNB5)"&�������������������������������������������������������')6BOTTTROIB;6)'#!!'��������������������MOW[hnytnrrjhf[XRNIM��������������������������������������������������������������������������������������������
#+/<HWbc_UH</ 
�������������������������/A@5)�������	!)DO[x�tk[6	)6AIQTPOLB96,) ��������������������������������������%)6=BDFCB6)	U[_gtuwtpphgg[YVUUUU;HMQLH;66;;;;;;;;;;;��������������������
##//;8/#
	��������������������������������W[cgt���������utga[Wpuvy}������������tqp���������������������������"#������� �������������/<UYZNHC:#
����IUa�����������xl^UMI��������������������)5B[gg[ULB;5)
qtw���������tjqqqqqq�� ���������t�������tqttttttttt�`�X�T�Q�O�T�U�`�m�m�m�j�`�`�`�`�`�`�`�`�
������
�#�<�U�a�m�a�H�/�&�-�#����
�����������������ĿѿԿݿ��ݿѿĿ������g�[�Z�T�V�Z�d�g�s�|�����������������s�g�5�����������5�A�N�p�z����u�s�g�Z�5�0�%�%�,�0�=�I�N�I�H�I�J�I�=�0�0�0�0�0�0�	���������������	���"�/�/�4�/�.�"��	¦ª­¬§¦�#��	����#�<�T�a�nÇÌÓÇ�z�a�H�/�#�н̽ͽν̽нݽ��������������ٽ������!�#�0�3�<�>�?�<�<�0�#��������վ��������ʾ�����	�����	����d�Y�O�G�F�M�X�m�������������������r�d�����������������������������þȾž��������u�����������"�;�F�J�Z�^�`�T�;���������L�J�L�V�Y�e�o�r�s�t�x�r�e�Y�L�L�L�L�L�L���������w�g�I�A�Z�s����������� ���	��ٿ��m�[�O�J�K�V�m�����Ŀݿ������%����$�����$�)�0�3�=�G�I�U�V�Z�V�J�=�0�$���������������������������������������
� �
���#�0�<�I�U�b�n�y�p�b�I�<�0��
ŠŜŔŒŏőŔŠŭų������������ŹŶŭŠ�����}�|�����Ľ��A�R�O�A�����ݽĽ��	����������	��/�H�T�d�c�[�[�U�K�;�"�	ùôù����������������������ùùùùùù�������������)�B�Q�X�S�O�B�6�)����������������������$�(�(�'�$�"����������������	��9�>�9�1������	��ÓÍÇÁÃÇÍÓàèìîðòìàÓÓÓÓ�G�E�G�O�T�`�m�n�m�`�_�T�G�G�G�G�G�G�G�G�лͻû»ûϻлܻ߻��ܻллллллл��s�r�n�n�x�����������������������������s�r�f�]�Y�V�Y�b�e�r�~�����������������~�r�ù����������ùϹܹ�����������ܹϹú3�,�1�3�@�H�L�W�T�L�D�@�3�3�3�3�3�3�3�3�������������������ĿȿѿѿѿĿ���������ƳưưƳ������������ƳƳƳƳƳƳƳƳƳƳ���	��������	����"�.�3�6�2�.�"�������z�x�y��������������������������������g�c�s�w����������������������������6�0�2�*�6�C�H�L�E�C�6�6�6�6�6�6�6�6�6�6čā�r�t�~čĞĦĳĿ������������ĿĦĚč�H�;�)�"��������	��#�/�>�H�P�[�f�f�c�H�����������������׾߾׾Ѿʾɾ������������H�;�6�7�9�9�H�T�m�z�������������z�a�T�H�!��� ���:�_�x�������������x�_�S�:�!�l�j�_�Y�S�Q�N�S�[�_�l�x�z�}�y�z�y�x�l�l���������������������������������������#�$��������������5�(��������(�5�A�N�P�T�R�N�A�>�5E7E/E*E*E*E*E7E8ECEPEPETETEQEPECE7E7E7E7�ּѼּؼ�����������������߼ּ��������$�0�=�V�b�f�m�q�o�b�V�I�0�ĿĳĦġėĕĝĞĦİĿ����������������Ŀ�C�6�*���������*�6�C�H�O�Z�\�U�O�C�I�<�0�#�!�������#�<�J�U�b�j�n�w�m�I�r�Y�M�B�>�B�M�r�������ɼ��мɼ¼����r�����������%���������������������ݿѿɿȿ����Ŀѿ�����6�;�1�����깪�������¹Ϲ���3�@�Y�c�]�L�'����ù��@�4�'���'�(�4�@�L�Y�f�r�u�n�f�Y�M�A�@Ç�z�m�d�^�Y�T�P�U�[�n�zÈÓÙáâàÓÇ�<�8�/�#�"�#�'�/�<�H�H�J�K�H�<�<�<�<�<�<��������!�#�.�4�9�.�!������²­²´¿����������¿²²²²²²²²² g r   % 5 > 8 7 v D b e 5 - Z [ g I 1 5 ~ 8 N 0 h b 5 h 7 f < 0 5 G j O 6 3 0 U J X ] ^ . ; R B 0  E | O 6 S I ; K A U p u F B 4    P    �  �  �  �  �  �  �  e  �    ~    w  �  �  U  X  �    �  �  �  �  l  W  y  �  \  9  �  B  d  �  f  0    �  �  u  6  �  �  P    )  �  P  �    �  g  �  �  �  B  �  |  �  O  W  z  �  6=C��t���`B���
�D����`B�49X�u�,1�D���D����j�\)��1��O߼�t����%�����+�49X������
��`B�<j�'m�h���o�t��m�h�,1�Y��0 ŽH�9�#�
�H�9��7L����49X������q��������^5��%�]/�ixս�j��1��o���w���㽍O߽��T�
=q��7L��F�C����T��/��vɽ����v�B�'B�Bm�B�B&B��A���B�B��B<�A��eB0��B��B4�@A��B��B%]B+V!B��B WB�1B�B%�&B		�B %�B�uB ��B	�PB�B ІB @�B�!B!֬B��B�WB��B�BXB��Bf�B�RB$qB�&BǲB�B��Bj�B	;A�cBbBb.B. MBl�B	��B
��B�%B�cB��B�.BZ�B*��B��B
?HB��B��B|<B��BS�B KBĂBTpA�x�B	WBBB?�A��B0��B��B4��A��}BN�B%�'B,0�B��B B~B��B��B'@EB�PB E�B��B �B	��B0B!A�B ?nB:?B!�5BA�B�^B�B9Bz6B@zB�SBBA�B?B��BF�B@DB��B	F!A�h�B�BA4B.B`BB	��B
D0BFHB�/BČB@�B�B*�B�B
@�B�MB�Ah@A�+CAwL�A��A��JB
�.A���A�l�A��WA-P�A��fAVI;@�\�AK�:A�R?���A��0Awk�B
n�A�1�A�A�}[A,�-A��A��Aָ�B��A[A�8�Ag�(@��%A�V�@[>���?�75At�qB��A]��A���A�D�B bFA�U}A���AKʇA�G�@���@��%A�KA��mA��sC��}A�eB
��A�X.A���A�M@�L�A1�AA���?S�*@��7AȼA�OZAqA�;�Ah�rA�~�Aw�A���A���B
��A� �A��)AA+(�A��AV�[@�-�AL�MA�|�?��A�y	Au�YB
�4A�Y^A좀A��A- �A�x�AϜ%A؁B�[A[A��Ah"�@�K~A�n�?��>�h?�ܭAt�B�sA]ُA�A���B ��A�~;A�AK�A���@�	�@��dA�PA���A�"C���At=B �A�bB �EA�UR@�ȓA1�A���?C��@�KMA�͑A�y�AG#A�|               2            &                  9      A   .      	         a   8               
            
      
               /      K   8      #   /            )                      L      9   I               	      !         )            !                  ;      A   9               ?   '            #                                 #   +      #   +      %   )         
                     !   /      '   1                                                            1      1   9               ;   !                                             #   '      !         %   )         
                        /      #   /               N �NK��NλkN��ZO�TN��O7SJN�<OkOf%N��8O��O�Y�N�c�P2�SN>y�PF��PqV�O��N��OۋO(sP���O�0NQ�KO��O��O+�eN�@�N%9N��OA�O%�N�!"N�DObN�N�'O�{�O�O�N@^IO�`�O��hO�P��P*m�N��sN�}�N=�rOs9GN��N_�<O�ʄO >"OG�Ozd�P.��Nn��O��P?@�N�~:OIQ�Ns��N�b�N*z    �  �  �  d  �  �  [  u  �  �  "  ,  '  �  �  �  �    *  �  �  �  �  �  �  �  q  R    A    �  �  �  �  E  �  U  �  �  �  $  :  �  )  �      �  	�  H  �  �  �  �  
%  �    �  �    �  D  �=t�%   ;o���
�e`B%   ��o�o��C��ě��o�49X�49X�49X���e`B��󶼓t����㼴9X��j���ͼ��0 ż����o�����'�h��h�����+��P�\)�t��t���P��P�#�
�,1�H�9�]/�<j�8Q�8Q�@��D���L�ͽT���m�h�e`B�e`B��%�q����o��%�}󶽏\)�����hs��1���罩�置{��������������������#-//166/#"?BN[cgjig[VNLB><????��������������������%+5BN[gqtul[NB5&! "%� $"������������	
	������������������������������������������������S[ht��������tpohec[S"$/1:;=;/""&*16CDOR\\YWOIC76*%&[`ht����������t[UQS[��������������������;Ham�������zaTH>722;Z[^hhmtxth[VTZZZZZZZ���#IUdnrqgC0#
��������������������
#/:<<<7/-# 
xz���������~zyxxxxxx����������������������%),))"��������#0Un~��{sbIB/���5BN[g�������t[NB5--5yz|�������}zyyyyyyyy�����������������������������������}z}��JNP[gt�������tkg[VNJ�������������������������������������������������������')6BOTTTROIB;6)'#!!'��������������������OZ[hjtutqkooh[ZUROKO��������������������������������������������������������������������������������������������
#'/<HO^_ZUH</
��������������������������'84)������%):JO[hsz|yth[OB) %")6?GOQOJBA6/)"��������������������������������������))6;BCEBB6,)#U[_gtuwtpphgg[YVUUUU;HMQLH;66;;;;;;;;;;;��������������������
#*/0/+#
�� �����������������������������Z[`fgt��������tga\[Zpuvy}������������tqp��������������������������� !������� ��������������
/<BIIE?6#
���Naz�����������yn^UMN��������������������)5BNVSNKB:5)qtw���������tjqqqqqq�� ���������t�������tqttttttttt�`�X�T�Q�O�T�U�`�m�m�m�j�`�`�`�`�`�`�`�`�<�1�/�.�/�<�H�U�Z�V�U�H�<�<�<�<�<�<�<�<���������������ĿʿпѿԿѿȿĿ����������g�f�Z�\�d�g�s�����������������s�g�g�g�g�5�(������(�5�A�N�b�l�t�r�g�Z�N�A�5�0�%�%�,�0�=�I�N�I�H�I�J�I�=�0�0�0�0�0�0�	���������������	���"�/�/�4�/�.�"��	¦ª­¬§¦�/�+�#������#�/�<�H�S�U�Y�U�H�=�<�/�н̽ͽν̽нݽ��������������ٽ������!�#�0�3�<�>�?�<�<�0�#����������پ˾ʾ����ʾ׾����	�����	���d�Y�O�G�F�M�X�m�������������������r�d�����������������������������þȾž���������������������"�<�G�H�D�3������������L�J�L�V�Y�e�o�r�s�t�x�r�e�Y�L�L�L�L�L�L�������������z�z�����������������������ٿ��m�[�O�J�K�V�m�����Ŀݿ������%����$�����$�)�0�3�=�G�I�U�V�Z�V�J�=�0�$���������������������������������������<�0�#�����#�0�<�I�U�b�n�v�n�m�b�I�<ŠŞŗŔŒŔŗŠŭŮŹ������������ŹŭŠ���������������н��-�A�L�I�A���ݽн��	���������	���"�/�=�G�L�K�H�J�D�;�"�	ùôù����������������������ùùùùùù���������)�B�O�Q�P�O�N�B�6�)������������������$�(�(�'�$�"��������	�������������	��"�+�(�"�����	ÓÍÇÁÃÇÍÓàèìîðòìàÓÓÓÓ�G�E�G�O�T�`�m�n�m�`�_�T�G�G�G�G�G�G�G�G�лͻû»ûϻлܻ߻��ܻллллллл��s�r�n�n�x�����������������������������s�r�f�]�Y�V�Y�b�e�r�~�����������������~�r�ù��������¹ùĹϹܹ����������ܹعϹú3�.�2�3�@�L�L�M�R�L�C�@�3�3�3�3�3�3�3�3�������������������ĿȿѿѿѿĿ���������ƳưưƳ������������ƳƳƳƳƳƳƳƳƳƳ���	��������	����"�.�3�6�2�.�"�������z�x�y��������������������������������n�j�s�z��������������� �������������6�0�2�*�6�C�H�L�E�C�6�6�6�6�6�6�6�6�6�6Ěčā�v�vĀďĠĦĳĿ������������ĿĦĚ�/�)�	���������	��"�/�6�C�I�K�O�P�H�;�/�������������������ɾʾ˾ʾž������������H�;�6�7�9�9�H�T�m�z�������������z�a�T�H�!��� ���:�_�x�������������x�_�S�:�!�_�[�S�S�P�S�]�_�l�x�x�{�x�w�x�y�x�l�_�_���������������������������������������#�$��������������5�(��������(�5�A�N�P�T�R�N�A�>�5E7E1E,E.E7ECECEDEPERESEPENECE7E7E7E7E7E7��ۼ�����������������������������$�0�=�V�b�f�m�q�o�b�V�I�0�ĿļĳĦĥĚěĦĦĳĻĿ��������������Ŀ�C�6�*���������*�6�C�H�O�Z�\�U�O�C�I�<�.�#��
�	���#�0�<�I�Q�Y�b�i�b�U�I�r�Y�M�C�?�C�V�r�������üڼܼͼƼ������r�����������%�����������������������ݿտпͿͿʿѿݿ�����1�7�,�������������ùϹ���3�@�Y�b�\�L�-����ù��@�4�'���'�(�4�@�L�Y�f�r�u�n�f�Y�M�A�@Ç�z�n�e�_�Z�Y�a�n�u�zÆÓØàáààÓÇ�<�8�/�#�"�#�'�/�<�H�H�J�K�H�<�<�<�<�<�<��������!�#�.�4�9�.�!������²­²´¿����������¿²²²²²²²²² g > #  ( > 8 7 B D b \ 5 - W [ X I 1 5 o 3 K E h M 5 D 7 f < 0 5 M d O 6 3 0 T J Q A V . ; D B 0  3 j O / S K : K % P p \ F B 4    P  f  �    �  �  �  �  @  e  �  w  ~    e  �  �  U  X  �  �  v  �  �  �  z  W  �  �  \  9  �  B    o  f  0    �  �  u  �  �  L  P    �  �  P  �  �  �  g  Z  �      �  �  �  O  �  z  �  6  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+  D+    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  m  R  h  {  �  �  �  �  �  �  �  �  �  �  �  �  f    �  �  !  �  �  �  �  �  �  �  �  �  �  �  �  x  `  9    �  �  K  %  Z  �  �  �  �  �  �  �  �  �  �  h  8     �  �  /  �  y  #  �  �    7  S  _  d  b  ]  T  B  '  �  �  �  0  �  #     �  �  �  �  �  �  �  u  e  U  ?  (    �  �  �  �  �    J  y  �  �  �  �  �  �  �  �  �  m  W  ?  (    �  �  �  �  V  
  [  B  .      �  �  �  �  �  k  H  !  �  �  x  ;     �  �  5  D  A  >  0  h  t  q  \  ;    �  �  `    �  �  <  �  9  �  }  u  l  k  l  l  f  ]  T  J  =  1  (  %  #     �   �   �  �  �  �  �  }  v  p  j  c  ]  R  C  3  $    �  �  �  �  �  �        !        �  �  �  �  �  �  Z  (  �  �  �  q  ,    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  8  '                      �  �  �  �  �  �  c  ;    �  �  ,  \  �  �  �  �  s  R     �  �  K    �  �    E   k  �  �  �  �  �  �  �  �  �  �  �  �  v  k  _  c  l  v    �    R  �  �  �  �  r  2  �  �  [  
  �  ^    �  b  �  I    �  �  �  k  I  '        �  �  �  �  f  (  �  �  N  �  Z      �  �  �  �  ^  5    �  �  �  P  .    �  s     �   v  *    	  �  �  �  �  �  q  L  '     �  �  �  R  !  �  �  �  �  �  �    j  V  B  /        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  W  7  "    �  s  �  @     �  y  �  �  �  �  �  d  3  �  �  �  {  +  �  �    �  x  h  i  4  N  l  �  �  �  �  �  z  0  �  �    �    r  *  �  �  [  �  �  �  �  �  �           �  �  �  �  �  �    j  U  A  ]  Y  f  k  i  v  �  {  b  I  0    �  �  r  8  �  �  F  �  �  �  �  �  �  w  _  A     �  �  �  �  [  #  �  �  $  �   �  �      (  +  0  I  e  p  m  ]  D  &  �  �  �  /  �  �   �  R  Q  Q  F  :  *      �  �  �  �  r  U  8    �  �  �                 �  �                  �  �  �  �  �  A  =  9  6  /  (  !        �  �  �  ^  F  .    �  �  �        �  �  �  �    P    �  �  E  �  �  Z  �  �  �    �  �  �  �  �  �  v  h  Y  O  J  H  I  H  A  8  +    �  �  Z  �  �  �  �  �  p  R  1    W  e  U  E  1       �  �  �  �  �  �  �  �  �  �  �  �  �  _  1  �  �  �  M    �  �  C  �  �  �  �  �  �  �  �  �  w  \  <    �  �    6  �  �  K  E  6  '      �  �  �  �  �  �  �  �  �  �  �  �  t  e  V  �  �  �  z  k  Z  G  3      �  �  �  |  R    �  �  ~  H  U  O  F  =  :  "  �  �  �  m  ?      �  �  �  u    u  �  �  �  �  �  �  |  h  c  5  �  �  �  G  	  �  a  �  @  n  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  C    �  �  U  
�  
�  	�  	W  �  �  �  �    �  �              �  �  �  �  K  �  �    �  �  C  
   �    .  8  4  (      �  �  �  �  {  ]  ;    �  �  x  8  �  �  �  ~  `  A  $    �  �  �  �  �  �  f  -  �  �  �  �  8  )    �  �  �  b    �  r    �  W  �  �  �     �  :  ]   �  �  �  �  �  �  �  }  X  -  �  �  �  r  C    �  �  T  �  b         �  �  �  �  �  �  �  �  �  �  �  �  n  @    �  �    �  �  �  �  �  �  �  }  k  X  E  0      �  �  �  �  �  �  �  �  v  a  F  (    �  �  �  y  D  �  �  !  �    z  �  	l  	�  	�  	�  	�  	�  	n  	F  	  �  �    t  �  �    8  H  N  J  7       .  D  ,    �  �  �  �  �  �  i  L  0    �  �  #  �  �  �  �  u  \  ?    �  �  �  y  E    �  b  �  P  �  �  �  �  �  �  �  �  �  �  �  �  �  r  V  9      �  �  �  k  �  w  m  `  U  N  F  =  2  #       �  �  �  i  8    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  H    �  o    �  
   
%  
  	�  	�  	�  	�  	h  	  �  F  �  @  �  8  �    X  e  �  �  �  �  �  �  �  w  l  `  U  I  >  2  &        �  �  �  �  �  	    �  �  �  L  7  
  �  �  0  �  S  �  F  �    X  �  �  h  5  �  �  m  /      �  �  �  n  -  �  v  �  V  �  �  �  �  }  h  S  >  (    �  �  �  �  d  1  �  �  �  T  &  n  
  �  �  �  �  �  g  I  *    �  �  _    �  U  �  �  T  �  �  �  �  �  �  �  �  r  Z  B  '    �  �  �  m  A    �  D  <  2  %    �  �  �  �  �  k  O  5    �  �  �  �  �  �  �  �  �  �  �  _  <    �  �  �  {  R  '  �  �  �  s  C  