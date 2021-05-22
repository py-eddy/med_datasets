CDF       
      obs    N   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�E����     8  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��l   max       P��     8  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��C�   max       <ě�     8      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @F�z�G�     0  !T   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @v������     0  -�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @P`           �  9�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @�:`         8  :P   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       <���     8  ;�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��~   max       B4�     8  <�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�}�   max       B4�}     8  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�6   max       C��A     8  ?0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�   max       C��v     8  @h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          W     8  A�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          =     8  B�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          ;     8  D   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��l   max       Pf�7     8  EH   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�Q   max       ?�q�i�B�     8  F�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��Q�   max       <ě�     8  G�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @F�z�G�     0  H�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @v������     0  U    speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @L@           �  aP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @�ɠ         8  a�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B   max         B     8  c$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?vOv_خ   max       ?�p��
=q     `  d\         &      
         	      
            $      
   !   "   +               A               
      %                        %      ;      W            -                              %   
                                    
               	      ?   *N�7N��eO� )ON}N��/N4*N�y�N�'�OH|NvZ�O�gN� IN=��O�L9O�b�N�^!O�w�P2|�PO��N8�NK�N��lP��N��[O��O21�O߮O\INO�1P�zN��OE�HO�	oO{�NAE�M��lO�3�O�?�N7��P 5hNqlO�qOߒO!�WO���PWBOK9�O��Oy��N���O��N�O7|OͭOq��O旫O֫O��N���N>O��O�"N�MN��N6MLN��N�L�Og�N�z�N�&�N��N�Nvf�N�O&�Ov��N���<ě�<D��<D��;�`B;ě�;ě�%   �D���D���D����o��o���
��`B��`B�t��#�
�49X�49X�T���u�u��o��o��t����㼣�
��1��9X��9X��9X��j��j�ě��ě����ͼ���������/��/��/��`B���������o�o�+�+��P����w��w��w�#�
�#�
�,1�,1�,1�49X�49X�8Q�8Q�8Q�@��Y��]/�aG��aG��m�h�q���u�y�#�}󶽁%�����C�����������������������������������������
#/3<RULH/#
*/4<HQU[_``UHC</&%**+/8;<HKTPH;/.*++++++#(0<><;20#�����

���������������������������������������������"#/4:<?</#""""""""kmqz���������zrmjhkkS\houwuh\QQSSSSSSSS��������������������(5BN[konkg[NF50���)1A=)��������������������������� ���������JTax����������zmaTHJ����������������������������������������?BIOQSSTOCB>????????OO[\ghsthd[TSOOOOOOO)5;BIDB=;52)$���������,,#�����GHSTaamrxrmaTHHEGGGG��59=;5)��������� #%&,/=JI<#
����������������������"/8;=>=;5/("35ABBENQNIB53.333333[g���������������~g[?BCIOO[_hihe\[OB????������������������6?FZhtyxsqh[OB)������ ����������������������������������������������gt������������tg_]_g)5BNZNKJD?5)
#+&#
�� %5@BA�����������������������������������
	��������Ngw�����������t[NKJNlt��������������tqkl��������������������09N[s��������g[N:/-0[_gtz�������tokdbe[[���
#/<HD@/,#
����HKUanz�����njaUOPKHH #%08<IIIHC<;0#    ��������������������<BNT[glgfc[NB=<<<<<<AGKNQ[bgoprqmg[NKBAAY[[egit~�������tg[ZY������������������������������������������
#,/6<</&#
���������������������|�������������������������	

�����������5IUbfihjgbUQ<0#��������������������)))1+)/6<BOTTQPOB761//////Z[bhtutpmh[WZZZZZZZZ##$*04<<C<0#"####��������������������/<UafmnswndUH<5/.,*/|�������������|||||| )469<<86+ qt{������������tpllq��

���������������

�������������������������������������������|{|�� /10-$#
������������������������������ּּּܼ����������������#������#�/�<�@�H�T�L�H�<�/�#�#�#�#ŹųŲŷ�������������������������Ź�A�8�4�0�/�4�7�A�M�Z�f�h�f�f�f�b�Z�M�A�A�����������������
����
���������������/�.�'�"�� �"�+�/�;�=�>�;�1�/�/�/�/�/�/��ÿùñòù�����������������������������$�������$�0�4�=�<�3�0�$�$�$�$�$�$ƧƥƚƗƒƚƤƳ��������������������ƳƧ�<�;�3�<�H�O�U�V�`�a�U�H�<�<�<�<�<�<�<�<ÓÍÇ�}�|ÀÇÓÚàìùþÿùôìàÓÓ�����������������������������������������������������ʾ;ξʾž��������������������������~���������������ȿȿĿ����������*�*�/�7�:�E�H�T�a�z�������z�m�a�T�H�;�*��s�u����������������������������N�G�N�N�S�V�[�g�s�����������������s�g�N��ƧƖƁ�h�d�c�uƆƚƳ��������� ������������ſŵůŶ�����������6�H�N�B�*����f�b�^�c�n�s������������ľž�������s�f�S�J�F�D�F�S�_�l�x�l�k�_�S�S�S�S�S�S�S�S�Y�X�W�Y�e�f�r�~��~�}�r�e�Y�Y�Y�Y�Y�Y�Y�����������������������������������������������r�d�Z�e�}���ʽ�!�.�9�8�!��ּ����Z�V�N�N�M�N�X�Z�g�s�w�}�|�s�s�g�Z�Z�Z�Z�X�O�Q�N�O�V�Z�g�s�����������������s�g�X¿´²«¡¦²¿����������������������¿�����������������������
������������������������������������������������������׾־׾�������������������������׾˾ʾϾѾ־�	�.�5�7�3�#���	���G�<�;�.�"�"��"�,�.�4�;�G�I�O�S�G�G�G�G�y�u�m�i�`�\�T�M�L�T�`�m�y�������������y���ڹϹù������ùϹܹ��������������M�@�$�����'�@�M�Y�f�r�z���v�f�_�Y�M��������������������������������A�;�>�A�C�M�N�O�M�A�A�A�A�A�A�A�A�A�A�A�i�^�T�Q�O�T�m�z���������������������z�i����ùöòóö�������������$�"������ҽ��������������������������g�=�)�5�A�Z�s�������������������������g�0�#�0�<�>�I�U�^�^�U�I�<�0�0�0�0�0�0�0�0�r�f�`�`�d�q���������ʼ��޼ּǼ�����r����������������������#�.�3�1�-�)���������������������������û»������������q�e�L�J�F�L�Y�r���������ɺк��������~�q�/�#�������#�/�H�Y�h�j�s�u�r�a�H�<�/££¦²����������������¿²¦£�����������������������������������������;�/����#�/�;�H�T�X�_�a�l�s�m�a�T�H�;�������������������������������������������������������������������ûȻлֻлû������������������������������������������*��� ������*�6�C�O�T�U�S�O�C�6�*ŭšŠŔōŇ�}�{�{ŇŔŠŭŹſ��Ż��Źŭ�G�3�.�(�"�'�.�;�G�T�`�m�v�����y�m�`�T�G�M�B�:�B�R�Y�hčěĦĴĽ��ĿĳĦĚā�h�M�������������������ÿĿȿѿӿؿ׿տѿĿ��������ɺ�����-�F�U�[�Y�S�-���ֺɺ������~�|�~�������������������������������ĽĽ��������ĽʽнڽнŽĽĽĽĽĽĽĽĽ����{�x�~�������ݽ����н��������������Ŀ����������ƿѿݿ�����������ݿؿѿľZ�T�M�L�M�Z�f�i�q�f�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z��������'�4�8�<�4�3�'�������������������	����������������������������ûлԻܻ��޻ܻлû����������I�G�B�B�I�V�b�o�{ǅǈǈǈ�{�o�b�^�V�I�IFF	E�FFFFF$F1F3F=FGFHFFF=F5F1F$FFF=F3F9F=FJFSFVFcFnFjFcFVFVFJF=F=F=F=F=F=�����ݽнĽ��������Ľнݽ����������h�e�h�k�t�x�~āčĖĜģĚđĐĕčā�t�hĿĶĵĻĿ��������������ĿĿĿĿĿĿĿĿ�����ܺ����������	���������
�������������	�����$�(�0�2�0�$��
���������������ʾʾ׾����׾ʾ�������EEEEE!E*E7ECEPE\E`EoExEwEiEPECE*EE���������������������������������������� 5 H K , * c " A a , 7 F C G ? E 7 > G ) V k Q W $ / h B ^ : P O [ c L d j / D R 3 h ; \ 2 k & A T , R W N 5 E F > Z � ' 3 p B f * b j O ) e e p % S < ? H '  #  �  P  Q  �  �  �  �    |  2  �  d  J  b  �  �  7  �    \  �  �  �  �  �  �  t  �  d  �  �  �  �    P    �  �  f  �  �  ,  l  [  �  n  �  p  �  �  �  �  �  R  �  5  [  �    0  �  S    �  Q  �  $  �  �  1  .  �  �  .  F  !  �<���%   �����o��o;o�T���D����o�T����C���`B�t��,1��󶼓t��,1�8Q�]/��㼛����ͼ�`B���
���ͽ<j�,1�8Q�o�ě��m�h��/�C��T���,1��`B��h�aG���o�+��{�t���h�@��@��D�����-�49X�q���T����w�Y��0 ŽT���@���+���-�T����\)�aG��H�9��hs��o�@��m�h�Y��m�h������罅���o��hs���������\)���-�o��;dBG,B��B�B�A���B%s�B �Bm[BsB"�A���B2�B4�B�B?�B�B��A��1B��B!�HB�BBb3B��B-
�A�?BՁB��BU&A��~B��B0NB�zB*�B��B"�7B)��B:�B
��B��BAtB�BQ+BG�B
B,�B �}B	@�B	�B4`B�B%�B>�Bm�B��B	��B��BABO�B�yB�NB#��B&�KB�4BBBC`B�+B%��B��Bq�B\B$�B
H�B�=B#��B�YBUB�BNBA�B�bB��B6yA���B%��B2rB��B�SBǮA�|�B2�B4�}B88B@wB1�B�A�~�B��B!�kB!BMB�6B-RA�v�B	BA�BBDA�}�B^ZB��B>�B*I&B?�B"�GB)��BC*B
��B�fB?wBP�B>�B}�B
�0B BB ��B	<�B
=�B��B��B%�PBF	B�SB�>B	�oB��BPB��B<]B�dB#��B&��B��B��B:�B�eB%I�B$IBCeB>�BB\B	�vB��B#��BM'B?�B>]B��AU�A�ISA���A=lA�ZQA�!�A���B	��B�A�� A�H�AKPyAN�cAsnA��AG�(A�.Bh�A�xvAH�_@��?풆A��0A �A��A���A���AлA�N�AV�/AX�Ab�sAlFu>�6@��A�;�A;�gA��AёA.xA�?A�
�@�a9A���@�mZ@	4�A��A�e7A�E�A���A�*@�5xA�k,B 	bA���Af�sA�LAx�0@j~@H?A'g�A%��A{�A?�@�=A@�m @��XBM�C�ƿC��AA*��A�+�A�,�@Tl�B�YAOMAC���A���AA�m]A���A<0A�{�A�H2A�(B
?�B8A�i�AˋDAKWAN�1As��A�z�AGFZA�n�B�4A�zAJ4a@�j?���A��AMA��wA�}�A���A�}�A�y�AW -AZ0�Aa�Al��>�@�FA�k-A;
YA�}�AғgA.�aA�m�A�v�@���A�w�@��[?�1�A���A�ZrA���A��qA�T�@��A�iB 3=A��Ag4A܄�Ax��@l�B@��A'�`A �A|��A?�h@��@��@�aoB��C��,C��vA)�6Aܚ A�{�@S9�B��AM,�C��A��{         '               
                  $      
   !   #   ,               A                     %                        &      ;   	   W            .                        	      &                                                      	      @   *                                                      -   )               =                     +         #         
   !         )      #   !      #   '                              #      %         '                                                                                                      +                  ;                     %                  
                  !         #   %                              !               %                                                N�7NkIO� )O�HN��/N4*N�4 N�'�O/|wNSK�O�gN� IN=��OA�O��=NFB�O�\P*%�O��OT��N8�NK�N�L]Pf�7N��Oa��N�t�N��QO\INO�1P�nN��O*�O�OO{�NAE�M��lO�,�O[�AN7��O���NqlO�;�O�;�O!�WO���O�U7OK9�O�m2OD��N���N��fN�O#Y�N���Oq��O޻�O֫OIMN���N>O�C|O�"N�MN��N6MLN9��N�L�OY&,N�z�N�&�N�T�N�Nvf�N�O&�N�5GN6�  �  �  �  �  �  4  �  7  �  �  7  �  �  �  �  �  U  �  �  \  �  �  �  �    �  s  �    f  M  �  �  �  E  �  5  �  �  �  y  �  R  A  �  F  �    �  :  a  �  �  z  r      �  7  �  |  C  �  �  v  �  �  K  {  �  M  E  �  (  �  �  I  �<ě�<t�<D��;ě�;ě�;ě��D���D����o��o��o��o���
�u�t��D���e`B�D��������C��u�u��t����
�����/��`B��`B��9X��9X�ě���j�ě������ě����ͼ�����h�\)��/�L�ͼ�`B�8Q�+������w�o�C��t���P�0 Ž�w�#�
�#�
�#�
�',1�Y��,1�49X�8Q�8Q�8Q�8Q�@��]/�]/�e`B�aG��m�h�u�u�y�#�}󶽁%��Q콬1����������������������������������������
#/3<RULH/#
+/6<HOUZ^^UH</(&+++++/8;<HKTPH;/.*++++++#(0<><;20#���

���������������������������������������������#$/29<<</#kmqz���������zrmjhkkS\houwuh\QQSSSSSSSS��������������������'5BN[egjige[NB<5,)%'��)+6=;1&���������������������������������������Taz���������zmaTMHKT����������������������������������������?BIOQSSTOCB>????????OO[\ghsthd[TSOOOOOOO')5A;95)))���������*(����MTamnuomaTKHMMMMMMMM��),5760)�����

#'/391/$#




��������������������"/8;=>=;5/("35ABBENQNIB53.333333ht����������������kh?BCIOO[_hihe\[OB????��������������������6@GO[htxroh[O6)������ ����������������������������������������������ag�������������}tfba)5?AACB@>75)
#+&#
�����'+*#������������������������������������������R[g{����������tg[PORlt��������������tqkl��������������������27AN[gt�������g[N@42[_gtz�������tokdbe[[���
#/<D?9/+#
�����NUanz�����znfaUUNKKN #%08<IIIHC<;0#    ��������������������<BNT[glgfc[NB=<<<<<<ABHLN[_gmoppkg[NLCBAZ[]ggkt�������tjg[ZZ������������������������������������������
#,/6<</&#
������������������������������������������������	

�����������6IUbhihihfbUQ<0#��������������������)))1+)/6<BOTTQPOB761//////Z[bhtutpmh[WZZZZZZZZ!#(028:0$#!!!!!!!!��������������������+/<UaelnqonbUH<5/.,+|�������������|||||| )469<<86+ qty���������tqmlqqqq��

���������������

�������������������������������������������|{|
#$'%#
 ������������������������ּּּܼ����������������#����� �#�)�/�<�D�@�<�/�#�#�#�#�#�#ŹųŲŷ�������������������������Ź�A�:�4�1�0�4�8�A�M�Z�f�e�e�`�Z�M�A�A�A�A�����������������
����
���������������/�.�'�"�� �"�+�/�;�=�>�;�1�/�/�/�/�/�/ùóõù��������������������ùùùùùù�$�������$�0�4�=�<�3�0�$�$�$�$�$�$ƳƧƚƙƖƚƧƨƳ��������������������Ƴ�H�<�<�6�<�H�U�U�U�^�_�U�H�H�H�H�H�H�H�HÓÍÇ�}�|ÀÇÓÚàìùþÿùôìàÓÓ�����������������������������������������������������ʾ;ξʾž����������������������������������������¿ÿ��������������;�0�:�;�A�G�H�N�T�a�m�z�������z�m�a�H�;��~�|��������������������������g�Z�R�V�X�Z�^�g�s�������������������s�gƩƘƁ�h�h�uƊƧ��������� �����������Ʃ���������������������'�*�,��������߾�s�g�i�t����������������¾�����������S�J�F�D�F�S�_�l�x�l�k�_�S�S�S�S�S�S�S�S�Y�X�W�Y�e�f�r�~��~�}�r�e�Y�Y�Y�Y�Y�Y�Y�����������������������������������������������r�`�b�s�����ʼ��!�.�5�5� �
�ּ��Z�S�P�Z�Z�g�s�t�{�z�s�g�Z�Z�Z�Z�Z�Z�Z�Z�g�_�Z�W�V�Y�Z�b�g�s�����������������s�g¿¿´µ¿��������������������¿¿¿¿¿�����������������������������������������������������������������������������������׾־׾�����������������������׾ξξҾؾ����	�+�3�5�0�!���	���G�<�;�.�"�"��"�,�.�4�;�G�I�O�S�G�G�G�G���y�m�k�`�]�U�O�O�T�`�m�y�����������������ܹϹù��������ùܹ����������������M�@�$�����'�@�M�Y�f�r�z���v�f�_�Y�M��������������������������������A�;�>�A�C�M�N�O�M�A�A�A�A�A�A�A�A�A�A�A�z�n�X�S�S�Q�T�a�m�z�������������������z����úý������������������������Ž����������������������������s�g�X�L�H�N�g�s�����������������������0�#�0�<�>�I�U�^�^�U�I�<�0�0�0�0�0�0�0�0����r�i�g�k�y�������żʼԼּԼͼ����������������������������� �+�.�,�'����������������������������û»������������q�e�L�J�F�L�Y�r���������ɺк��������~�q�<�/�#�������/�<�H�U�b�n�o�l�a�H�<££¦²����������������¿²¦£�����������������������������������������/�#���'�/�;�H�T�T�X�\�a�f�b�a�T�H�;�/�����������������������������������������������������������»������������������������������������������������������������6�*���
�����*�6�C�O�S�T�R�O�C�:�6ŭŧŠŔŒŇ�ŁŇŔŠŭŹż��ŹŶŵŭŭ�G�3�.�(�"�'�.�;�G�T�`�m�v�����y�m�`�T�G�O�B�;�B�S�Z�hāĚĦĳķļľĳĦĚā�h�O�������������������ÿĿȿѿӿؿ׿տѿĿ�����������!�-�=�F�J�F�A�:�-�!�����������~�|�~�������������������������������ĽĽ��������ĽʽнڽнŽĽĽĽĽĽĽĽĽ����|�y�~�������ݽ������н������������Ŀ����������ƿѿݿ�����������ݿؿѿľZ�T�M�L�M�Z�f�i�q�f�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z��������'�4�8�<�4�3�'�������������������	��������������������������ûлܻ�ܻܻлû����������������I�G�B�B�I�V�b�o�{ǅǈǈǈ�{�o�b�^�V�I�IFFF
F FFFFF$F1F1F=FFFGFEF=F4F1F$FF=F3F9F=FJFSFVFcFnFjFcFVFVFJF=F=F=F=F=F=�����ݽнĽ��������Ľнݽ����������h�e�h�l�t�{ĀāĉčĎĎēčā�t�h�h�h�hĿĶĵĻĿ��������������ĿĿĿĿĿĿĿĿ�����ܺ����������	���������
�������������	�����$�(�0�2�0�$��
���������������ʾʾ׾����׾ʾ�������E*E%E(E*E5E7ECEPERE\E^EaE\EPECE7E*E*E*E*���������������������������������������� 5 < K ) * c % A [ . 7 F C 5 9 C ) ; 5 $ V k : T $ 2 _ ! ^ : H O Y h L d j ) R R ) h 5 V 2 k & A O $ R C N - F F 8 Z \ ' 3 q B f * b _ O % e e P % S < ? & '  #  p  P  5  �  �  �  �  �  d  2  �  d  �    n      ;  �  \  �  �  U  �  �  �  �  �  d  �  �  �  i    P    V  �  f  T  �  �  �  [  �  
  �  `  �  �  �  �  b    �    [  h    0  �  S    �  Q  h  $  �  �  1  �  �  �  .  F  �  F  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  �  �  �  �  �  r  `  O  =  +      �  �  �  �  �  �  �  p  u  u  y  �  �  �  �  �  �  �  {  \  2    �  �  Q    �  Y  �  �  �  �  �  |  c  F  #  �  �  �  ^    �  x    [  w  �  �  �  �  �  �  �  �  �  x  b  J  3    �  �  �  �  t  O  0  �  �  �  �  �  �  �  �  o  X  0  �  �  �  M    �  �  �  Z  4  /  *  %        
    �  �  �  �  �  �  �  �  �  x  e  �  �  �  �  �  �  �  �  �  �  �  �  �  j  A    �  �  �    7  $    �  �  �  �  �  �  u  \  B  '    �  �  �  �  �  n  q  ~  �    v  k  _  P  ;  "    �  �  �  �  V    �  _   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  b  O  ;  '    7  ,      �  �  �  �  �  d  7    �  �  r  R  -  �  �  �  �  �  �    {  x  t  o  j  d  _  Y  T  N  I  C  >  8  2  -  �  �  �  �  ~  x  k  _  R  E  4       �   �   �   �   x   W   6  6  v  �  �  �  �  �  �  `  :    �  �  _    �  P  �  J  �  �  �  �  �  �  �  r  _  I  4  $    �  �  �  m     �  8   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  C  R  U  P  F  6  #    �  �  �  �  �  c  .  �  q  �  u  �  �  �  �  �  �  f  F    �  �  �  j  /  �  �  l    �   :    i  �  �  �  �  �  �  �  �  �  �  2  �  ~    �  B  x   �  ;  H  V  \  M  :  $    �  �  �  |  U  )  �  �  J  �  �  �  �  v  h  Y  K  <  -      �  �  �  �  �  �  �  �  �  �  {  �  �  �  �  �  t  \  F  0      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  �  �  }  q  c  T  D  3  h  �  �  �  w  n  `  B    �  �    R    �  0  �  �  "  ]              �  �  �  �  �  �  �  �  �  �  j  N  0    `  l  t  |  �  �  �  ~  o  \  @    �  �  �  R    �    N  �      3  W  l  q  s  h  R  7    �  �  �  <  �  �  �  l  �  �  �  �  �  �  �  �  �  �  �  �  n  ;  �  �  .  �  �  F    �  �  �  �  �  �  �  v  m  _  K  0    �  �  �  d  0   �  f  ^  V  M  E  <  4  ,  #         �   �   �   �   �   �   �   �  :  L  D  8  &      ,  =  D  E  6    �  �  �  c  �  �  �  �  �  �  w  l  `  R  D  6  )         �   �   �   �   �   �   l  �  �  �  �  �  �  �  �  }  o  `  O  =  ,    �  �  �  a  -  �  �  �  �  �  �  �  �  �  �  o  G  %  &  H  (  �  �  �  ?  E  /      �  �  �  �  �  g  I  '      �  �  �  k  ,  �  �  �  �  �  �  �  �  �  �  �  �  �  |  r  a  P  ?  .      5  3  2  0  .  -  +  &               �  �  �  �  �  �  �  �  �  �  �  �  Z    �  �  n  V  L  d  ~  V    �  i    �  H  �  �  �  �  �  �  �  ~  g  N  ,  �  �  �  N    �    �  �  �  �  �  �  �  m  O  2    �  �  �  b  1      �   �   b  -  K  X  ^  ]  `  l  v  x  i  @    �  z    �  �  [  �  '  �  �  �  �  �  |  c  I  8  (    �  �  �  �  �  y  l  q  v  
�  
�  -  Q  M  2    
�  
�  
`  
  	�  	�  	�  	Y  �  �  �  �  p  <  >  @  9  *      �  �  �  �  _  =    �  �  �  �  Y    �  �  �  �  �  }  j  W  ?  $    �  �  �  h  8    �  �  �  F  4  !    �  �  �  �  t  I  4    �  �    U  /  	  �  �  r  �  �  �  �  �  �  m  ?    �  �  F  �  �  ,  �  H  \  m              �  �  �  �  �  �  �  i  A    �  |     �  �  �  �  �  �  |  o  ]  M  =  *    �  �  �  J  �  �  �  �  .  (  #  :  .      �  �  �  ~  S    �  �  Z  �  �  z  5  a  ]  Y  V  R  N  K  G  C  ?  :  4  -  '                \  l  t  y  |  {  �  �  }  s  e  R  ;     �  �  �  9  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  c  L  0     �   �   �  u  x  v  n  _  Q  A  /      �  �  �  }  Y  0  �  �  �  ^  _  g  o  q  o  i  [  L  8  "    �  �  �  �  �  j  I  !   �    �  �  �  �  �  �  �  �  �  �  z  ^  5    �  �    �          �  �  �  n  <    �  �  :  �  �  &  �    n  +  i  �  �  �  �  �  f  K  .    �  �  �  �  �  �  U    �  �  A  �  �  �  �  �  �    -  0    �  �  �  W    �  �  M  �  �  �  �  �  �  �  �  �  �  �  r  R  /    �  �  �  z  U  1    |  v  o  h  `  O  ?  .    	  �  �  �  �  �  �  n  R  6    9  @  *    �  �  �  �  t  V  ^  L    �  n  $  �  �  <    �  �  �  �  �  �  l  O  0    �  �  �  t  9  �  �  I  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  x  q  k  d  v  c  O  9  !    �  �  �  �  _  ;    �  �  \  �  �  @  �  �  �  �  �  �  �  �  �  �  v  [  :    �  �  Y    �  `    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      K  @  4  $      �  �  �  �  �  �  �  s  ]  G  :  8  <  D  [  t  Z  7    �  �  �  s  ?  �  �  X  �  �  I  �  j  �  B  �  �  �  �  �  �  �  s  b  I  .    �  �  �  x  M  #  �  �  M  @  2  %          �  �  �  �  �  �  �  �  �  �  �  �  �  '  C  <  >  A  B  5     
  �  �  �  �  �  v  c  O  .    �  �  t  ^  G  ,    �  �  }  /  �  �  ,  �  Y  �  i  �  f  (  %  #             �  �  �  �  �  �  �  s  V  8     �  �  �  �  �  �  n  Z  F  7  )    �  �  �  �  �  �  g  J  -  �  �  �  �  �  �  l  Q  9      �  �  �  w  K  �  �  �  �  
�  
�    J  �  �  )  >  H  =    �  G  
�  
J  	�  	&  q  _  �  �  �  �  �  (  �  �  �  �    �  3  f  
�  
�  	�  �  �  �  `