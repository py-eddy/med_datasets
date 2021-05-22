CDF       
      obs    E   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?���vȴ:       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N
Z�   max       P�U@       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       <�`B       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?p��
=q   max       @F�p��
>     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ᙙ���    max       @v`(�\     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @P            �  6x   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�         max       @�*�           7   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �
=q   max       <�j       8   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B4o�       9,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B4w       :@   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >q�   max       C��.       ;T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >~~�   max       C��i       <h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          h       =|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;       >�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;       ?�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N
Z�   max       P�<+       @�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�4�J�   max       ?���       A�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       <�`B       B�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?xQ��   max       @F�p��
>     
�  C�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ᙙ���    max       @v`(�\     
�  N�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @P            �  Y�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�         max       @�*�           Z   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�       [$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?|�1&�y   max       ?���     �  \8         
                  	         -         ,                       7         Z               *            h               I            $   0   3               	      	                      	   1                  %      	   ,N
��Os�;N�4GO^�oO0ٜN(CON�N<��N:k�O�~N�>�PF<O�}�OuL�P�<+N���O
6�N�ߏO^�O�LO~�P@�[N-�'N폗PwO�v=N�u}Od�O�2O�ͷOA	�N��ND�P�U@OtNmK�O�N
Z�PKL�Op5�N�t�NmMP$lP ʢO�Z�NM�:Oq�N���N��N�WN�UeN��N��OJ�O�#*O�s�Oy�N���N���O�&�O»OدN���N� O8�O�ݱO�NO�hO2St<�`B<D��;ě�;�o:�o:�o�D���D���e`B�u�u��t���t���t����
��1��1��1��9X��j��j���ͼ��ͼ�������������`B��h��h���������o�o�+��P��P��P�'',1�8Q�<j�H�9�L�ͽL�ͽP�`�T���T���T���T���T���T���]/�m�h�m�h�m�h�}�}󶽁%�����O߽�\)��\)������㽟�w���� )**)��
#*0B<0#
�������)/5<?@5)gnpz�����������zqncg9<DHUanyz|}zvnaUI<:9��������������������������������������������

��������������������������������!)-,)&���itz����������~ywvwtiEDGK[t������pqgNB<=E��������������������

#/<EFHOM</#
�
#0In{����{N<0���MNP[gqtztog][RNJMMMM3<EHUaenkgaUH<933333��������������������>CO\huv{urlh\ONCB<>>5BKNNJB5)�����������������������������������������MOQ[hmjh[OMMMMMMMMMM��
#+/30/#�����������������}|}���������������������$)5BDNVNB<5.)#$$$$$$#$/<HU]aXUHH</%#������������������������05EHB5)����&6BDEFFB6)558@@BENQRRPNB:55555@BOX[hlh[[OJBA@@@@@@u�����������xsu��� )+1-)	�������������������������������������������������������0IUbkmg_UI0#
�����HLPTZahmsyzyumaXTIHH��������������������
#+#
	>BPg����������t[NB9>�����������������~~�GUahnz�����znaUJGEG������������������������
#
�������{{������}zvnicnx{{{{QTVZ_amnnjjjigaTQNNQ���������������������������������������������������������������������������)/6COQOF60)�����������������~�����������������~������

�������������	������������������������������������������������NNQX[gtx{wtphg[SNLNN�����
!%'"
�������=BCNYYYNKB>=========����������������������������������������)5DNgt{tg[NKB;)!)Udln{|����|{nb]ZUUU����������������������
"$$! "
����ɺƺǺɺԺֺ����ֺϺɺɺɺɺɺɺɺɼ����������������ʼ˼Ҽ����ּʼ������I�H�=�1�0�.�0�4�=�I�V�^�_�Z�V�J�I�I�I�I������������������������
������������������������������������������������ؾ����������������������������������������M�H�C�F�M�Z�\�s���������������s�f�Z�M�(�����!�(�0�4�9�:�4�(�(�(�(�(�(�(�(�t�l�g�d�d�g�t�~��w�t�t�t�t�t�t�t�t�t�t�3�,�'������%�'�3�;�@�I�Q�N�L�@�8�3�л̻лֻֻջܻ������������ܻٻпĿ������q�c�i�m�y�������ݿ�������ݿľ'�"�������(�4�A�Q�S�R�O�Q�M�A�4�'�U�J�H�<�-�(�%�/�H�L�U�a�n�{�}�z�n�a�W�U���������i�]�M�9�6�O�g����������������f�]�Z�R�V�Z�\�f�l�s�z�����~�s�f�f�f�f�����������	�����	�����������ܹϹȹϹҹܹ���������������;�8�3�0�2�9�;�G�O�T�`�h�i�a�`�U�T�G�;�;�����u�X�W�g�s���������������������������Y�N�N�Q�R�Y�f��������������������r�f�YŠŇ�{�h�Y�d�nŇŠŹ���������	������Š�Ϲɹù����ùϹҹ۹չϹϹϹϹϹϹϹϹϹ����
����#�)�0�6�B�D�K�I�B�6�)�!���'��"�,�@�Y�e�����ֺ���ܺɺ����Y�L�'�����y�k�c�k�m�t�y���������ɿֿ�ݿѿĿ������������	�������	�������������M�B�A�9�8�:�>�A�M�Z�\�f�f�m�o�f�c�Z�M�M�l�f�_�Y�W�W�\�_�l�x�|�������������x�l�l��� �	��#�;�G�a�m�����}�x�w�n�a�H�"�����������x�p�u�x�������������ɻû�����ŇŅ�{�n�b�X�b�i�n�{ŇŔŕŖŔŋŇŇŇŇ��������������	��	�����������û��������������л����@�Q�J�9���ܻür�l�f�[�b�f�n�r�}�������������������r����������������������������������������ÇÅÀÂÇÇÎÓàçìóõíìàßÓÇÇ�����������������������������������򽐽u�i�a�`�h�y���Ľ����߽߽۽н������������������������	��"�/�<�9�/�*��	���r�k�f�e�c�e�n�r�~�����������������~�r�r�лǻû��û̻лջܻݻ߻ܻллллллл��������������������$�3�8�9�4�0�3�2��������ƳƪƧƳ��������%�.�1�'�$������ù������z�y���������Ϲ��������ܹϹú��������źɺֺֺٺۺֺɺ����������������ɺǺ������ɺ����"�$�#�������ֺɿ����������Ŀ˿ʿĿ¿ĿǿĿ��������������g�\�Z�N�A�>�A�A�N�Z�g�s�������������s�gùùù������������������ùùùùùùùù�;�7�;�;�?�D�H�N�T�`�a�c�a�_�U�T�I�H�;�;àßÜÞàìíöñìàààààààààà�N�L�L�N�Z�_�g�s�s�����������t�s�g�Z�N�N������������������������������������������������������0�N�V�_�^�V�I�=�$���#���������#�0�I�Z�e�h�_�U�<�0�#�����������������
����#�*�)�#��
����������������!�)�.�5�0�.�!��������ĳĲĮĳĿ����������������Ŀĳĳĳĳĳĳ�6�&���&�6�O�[�h�tāĚĩĲĦĚā�h�O�6�
����������������
���#�$�)�#���
�
�Z�A�5�)�>�9�*�5�N�g���������������s�g�Z�a�U�U�K�U�a�n�y�z�}�z�n�a�a�a�a�a�a�a�aččăĂčĚĦĳĳĿ��ĿļĳĦĚčččč�)������)�7�B�O�[�h�j�t�h�]�O�B�6�)½º²ª°´¿�����	�������������½�ϻû��������ûлܻ����	�������޻ܻ�ƚƎƁ�u�o�u�xƁƎƚƣƚƚƚƚƚƚƚƚƚE*E EEE%E*E7ECEPE\EiEuE�E�E�EuE\ECE7E* Y A ! * 5 = + 1 o ) a g ^ + ] 2 4 | = H 3 C < a K Z R ? G a < V Z 8 I 8 B k 9 O 0 B D ; D 8 [ � ~ X | d O = f @ : R g \ = 8 A > R ) 8 T n    G    �  �  �     �  N  �  7  $  �  r  �  �  �  A  �  ?  3    �  R  ;  #  �  �  g  ]  b  �  �  �  �  F  �  V  A  s    �  @  �  l  �  m  (  �  `  R    \  �  �  W  }  k  �  �  }    �    �  �  
  X  h  �<�j�o�D�����ͼ��㺃o���ͼu��j��h��`B�}�t��'�o���\)��/�����]/�]/�����t��0 Ž�h���\)�aG��L�ͽ�t��<j�\)�C��
=q�D���#�
��+��w��G���\)�T���49X���T��������aG���7L�m�h�u�u�m�h�y�#��+��+���P��Q콗�P��o��hs��G�������^5������{�ȴ9��;d���-���mB�"B$t{B�WB)
B4B4o�B ��B#� B��BCVB�B	`B��B�5B&f�B�B|rB�+B1�LB�7B @B��BIJBFBB+6]Bf�BɻB!?�B�B^,BަB�B/TBR[B0�BH�B
��B%�bA���B!b�B$��B
�B�;B�B!5CB#�B��A���B�BI�B��B�2BQ$B��B��B�<B-�B ��BP�B	�By�B*�BN�B�jBj�B(K�B�%BՈB�B$�KB�B>�B9@B4wB �eB#��B�BC�B?�B	B��B��B&�
B�fB`B��B1ťB��B ?2B�BBB��B@B+A.B=�B�/B!>�B�WB@fB��B�QBB|B?oBD�B?�B
�$B&H_A��cB!DJB$��B
U�B��B:�B!�B#�BL�A���B�!B��B��B�&B?�B�8B��B-B.A0B �#B��B�B��B48B?B;BA�B(-�B�#B>�@;C~@���B6�A�u�A��CAK�<AA��A7��A���?��@���As�1A9��A�-RA���AA��AY?~�Ad�=A�\�@�POA��Q>��A�ӑ@AseAZ#�A=m@�m
A�EC@�R�A�aAX.�@��K@�Q�AI�lA��HBj~A"f�A���@O@�BA���B��>q�@4�M@L{2Ax<�A���A��A��A���A��ZA���B	A�y�A�<PA
�A�U�A��9A�?A���A��A�gA�L!A�1@�1�B��C��.@<h*@��BA�	A�|,AK�6AA4BA7�A���?�C�@�Ar��A9�pA�>A��KAAnAX��>�(:Af<�A�P@�u�A��h>~~�A��@�XAq8iAZ�gA=�@��A�fB@�+\A�AYw�@�[@��AH&@A�/$B�`A ��A���?�"�@��A�b�B�z>�ɓ@2�@?�Aw*A�o3Aά2A���A�q�A��|A��4B
�}A쁐A�yRA'�A���A��A�	UA��A��^A��A؆�A��@�"�BE�C��i         
                  
         -         -                       7         [               +            h               J            %   1   3            	   	      
            !         
   1                  %      
   -                                    )         ;               %      /         /   %            %            9               /            '   '   %                                             )      '            !                                                      ;               !      +         -   !            #            7               -            '   %   %                                                   !            !         N
��O*�N�4GN�x�O$�^N(CON�N<��N:k�N��N��O�O�}�N�P2P�<+Ns��N�$�N�ߏO^�O�]	OH~�P�&N-�'N�ɷPG�O�cN�u}Od�N��O�_OA	�N��ND�Py^�OtNmK�O�N
Z�P(5$O,ؕN�t�NmMP�iO�,O�Z�NM�:Oq�N���N��N�WNz�N��N��OJ�O�#*OAw�Oy�N���N���O���O»O�A�N���N���O8�O�ݱO�NO�hO2St    �  �  �  
  �  �  �  G  4  �  Q  �  $  ]  �  i  �  P  �  �  �  �  �  	  <    �  \  �  �  �  �    �  �  �  �  	    �  �  i  �  	�  b  �    �  �  �    '  X  �  u  �  �  �  �  |  �  �  �    �  C  �  �<�`B<o;ě��o%   :�o�D���D���e`B���
���
��㼓t������
��j������1��9X�ě���`B�����ͼ�`B����/��`B��h�+�+�������<j�o�+��P��P�8Q�<j�',1�@��D���H�9�L�ͽL�ͽP�`�T���T���Y��T���T���T���]/��+�m�h�m�h�}󶽑hs��%��C���O߽�hs��\)������㽟�w���� )**)�
#'-)#
�������)/5<?@5)����������{{��������;<FHUanxz{{zunaUKH;;��������������������������������������������

������������������������������'�����������������}{����������MNOW[gqtz|xtpg[SNNMM��������������������!#/0785/#!!!!!!!!�
#0In{����{N<0���MNOU[ggtutlg[NMMMMMM:<?HU^ada_UH?<::::::��������������������>CO\huv{urlh\ONCB<>>)5BJNKIB5)�� ����������������������������������������MOQ[hmjh[OMMMMMMMMMM�
#)/1/.#
	��������������������������������������������$)5BDNVNB<5.)#$$$$$$#$/<HU]aXUHH</%#����������������������.5CEB5)�����&6BDEFFB6)558@@BENQRRPNB:55555@BOX[hlh[[OJBA@@@@@@z��������������~xz��� )+1-)	��������������������������������������������������������#IUbgkdbUI<0#
����T^admosvxvsma[TRMKMT��������������������
#+#
	@Sg�����������t[NB;@������������������GUahnz�����znaUJGEG������������������������
#
�������{{������}zvnicnx{{{{QTVZ_amnnjjjigaTQNNQ���������������������������������������������������������������������������)/6COQOF60)�������������������������������������������

�������������	������������������������������������������������NNQX[gtx{wtphg[SNLNN���
!$% 
��������=BCNYYYNKB>=========����������������������������������������)5DNgt{tg[NKB;)!)Udln{|����|{nb]ZUUU����������������������
"$$! "
����ɺƺǺɺԺֺ����ֺϺɺɺɺɺɺɺɺɼ����������������ʼͼּ����ּʼ������I�H�=�1�0�.�0�4�=�I�V�^�_�Z�V�J�I�I�I�I���������������
�����������������������������������������������������������ؾ����������������������������������������M�H�C�F�M�Z�\�s���������������s�f�Z�M�(�����!�(�0�4�9�:�4�(�(�(�(�(�(�(�(�t�l�g�d�d�g�t�~��w�t�t�t�t�t�t�t�t�t�t�3�(�'�!�#�'�3�@�A�J�F�@�3�3�3�3�3�3�3�3�ܻܻܻܻ����������ܻܻܻܻܻܻܻܻܻܿ��������������������������ſͿĿ��������'�"�������(�4�A�Q�S�R�O�Q�M�A�4�'�<�;�;�<�H�U�a�i�h�a�U�H�<�<�<�<�<�<�<�<���������i�]�M�9�6�O�g����������������s�l�f�Z�V�Z�Z�d�f�g�s�w�}�v�s�s�s�s�s�s��������������	���	������������������ܹϹȹϹҹܹ���������������;�8�3�0�2�9�;�G�O�T�`�h�i�a�`�U�T�G�;�;�����v�b�Y�Y�g�s�������������������������Y�S�R�U�Y�f�r�������������������r�f�YŗŇ�n�d�p�{ŔŠŭŹ���������
�
�����ŗ�Ϲɹù����ùϹҹ۹չϹϹϹϹϹϹϹϹϹ������$�)�3�6�?�B�I�G�B�7�6�)�����@�3�+�.�4�@�L�Y�e���غ���޺ֺ����Y�@���������{�l�f�i�m�y���������ǿԿݿ׿Ŀ������������	�������	�������������M�B�A�9�8�:�>�A�M�Z�\�f�f�m�o�f�c�Z�M�M�x�p�l�_�^�[�]�_�b�l�x�y�������������x�x��
���&�;�K�a�m�z���z�u�u�l�a�T�H�/�����������x�p�u�x�������������ɻû�����ŇŅ�{�n�b�X�b�i�n�{ŇŔŕŖŔŋŇŇŇŇ��������������	��	�����������û������������û���4�<�C�=�4���ܻür�l�f�[�b�f�n�r�}�������������������r����������������������������������������ÇÅÀÂÇÇÎÓàçìóõíìàßÓÇÇ�����������������������������������򽞽��s�e�e�n�y�����ҽ�����ܽ׽ֽнĽ������������������	��"�/�7�6�/�&�"��	����r�k�f�e�c�e�n�r�~�����������������~�r�r�лǻû��û̻лջܻݻ߻ܻллллллл��������������������!�0�5�6�2�-�/�,����������ƳƬƯƳ��������#�-�0�$������ù������z�y���������Ϲ��������ܹϹú��������źɺֺֺٺۺֺɺ����������������ɺǺ������ɺ����"�$�#�������ֺɿ����������Ŀ˿ʿĿ¿ĿǿĿ��������������g�\�Z�N�A�>�A�A�N�Z�g�s�������������s�gùùù������������������ùùùùùùùù�H�>�@�E�H�P�T�]�a�b�a�]�T�Q�H�H�H�H�H�HàßÜÞàìíöñìàààààààààà�N�L�L�N�Z�_�g�s�s�����������t�s�g�Z�N�N������������������������������������������������������0�N�V�_�^�V�I�=�$���0�$�#����#�(�0�<�D�I�S�^�a�V�U�I�<�0�����������������
����#�*�)�#��
����������������!�)�.�5�0�.�!��������ĳĲĮĳĿ����������������Ŀĳĳĳĳĳĳ�6�3�0�)� � �,�6�B�O�[�h�pāČā�t�O�B�6�
����������������
���#�$�)�#���
�
�J�>�B�=�0�5�A�N�g���������������s�g�Z�J�a�U�U�K�U�a�n�y�z�}�z�n�a�a�a�a�a�a�a�aĚďčĄąčĚĦıĳľĹĳĦĚĚĚĚĚĚ�)������)�7�B�O�[�h�j�t�h�]�O�B�6�)½º²ª°´¿�����	�������������½�ϻû��������ûлܻ����	�������޻ܻ�ƚƎƁ�u�o�u�xƁƎƚƣƚƚƚƚƚƚƚƚƚE*E EEE%E*E7ECEPE\EiEuE�E�E�EuE\ECE7E* Y 2 ! 7 4 = + 1 o 0 ' 1 ^ ' ] ; 0 | = @ 9 D < Z H V R ? Q \ < V Z 9 I 8 B k ; G 0 B E : D 8 [ � ~ X o d O = f # : R g - = & A = R ) 8 T n    G  {  �  �  t     �  N  �  �  .  R  r  �  �  �  �  �  ?    �  
  R  �  �  n  �  g    �  �  �  �  !  F  �  V  A    �  �  @  �  :  �  m  (  �  `  R  �  \  �  �  W  �  k  �  �  _    |    �  �  
  X  h  �  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�            �  �  �  �  �  �  �  �  �  x  `  G  .    �  �  �  �  �  �  �  �  �  �  �  �  �  y  S    �  �  �  U    �  �  �  �  �  {  l  Z  G  3      �  �  �  �  p  R  4    j  �  �  �  �  �  �  �  �  �  �  �  �  �  Z  �  *  �  �    	  
      �  �  �  �  �  �  �  z  P    �  �  )  �  1  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  w  q  j  �  �  �  �  �  �  �    w  q  n  ]  G  -    �  �  �  L    �  �  �  �  �  �  �  �    z  t  o  i  Z  8     �   �   �   �  G  N  U  T  S  F  6       �  �  �  e  :    �  �  o  �  �  �  �      '  -  1  3  .  $      �  �  �  p  G  #  �  R  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j    �  D  �  P  `  }  �  �  �  �  �  �  !  O  7    �  �  �  ]  !  �  '  $  �  �  �  �  �  u  �  �  �  �  �  �  q  U  0  �  �  t  %  �  s  �  �  �  �  �        #  #        �  �  �  �  9  �  ]  7    �  y  #  �  �  P  A  $    �  �  �  k  %  �  S   F  �  �  �  �  �  �  �  �  �  �  r  W  5    �  �  �  w  M  #  O  V  Z  ^  b  f  h  h  d  Y  M  >  *    �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  s  k  f  a  `  j  u  ~  �  �  �  P  J  D  =  7  1  +  %              �   �   �   �   �   �   �  �  �  �  �  �  �  �  d  -  �  �  p  0  �  �  U  �  �  4  �  �  �  �  �  �  �  �  �  �  �  �  a  @    �  $  �    �    �  �  �  �  �    }  z  f  G  !  �  �  �  Z  �  @  �    �  �  �  �  �  �  �  �  �  �  z  p  f  \  Q  F  9  +    	  �  �  �  �  �  o  K  $  �  �  �  �  �  u  ?    �  y  :       �  �  		  	  �  �  �  n  '  �  �  S    �  V  �  f  �  �  o  4  9  6  )      �  �  �  �  �  �  }  i  Z  D  &   �   �   w        �  �  �  �  �  �  �  y  _  4  
  �  �  ~  J     �  �  �  �  �  �  �  �  s  S  &  �  �  p  .  �  �  V    �  8  D  P  X  [  [  W  J  /    �  �  y  O  .    �  �  m    �  �  �  �  �  �  �  �  z  y  T  2  
  �  �  ?    �  _  �  �  �  �  �  �  �  �  �  �  �  �  �  }  ^  1    �  `      �  �  �  �  �  �  �  �  }  {  x  s  k  c  [  S  b  z  �  �  �  �  �  �  �  �  �  �  �  �  w  l  `  U  I  ;  ,         �  
�  
�  	    
�  
�  
�  
g  	�  	�  	  �    �  �  B  �  �      �  �  �  �  �  �  �  t  b  O  4    �  �  }  R  :  1    b  �  �  �  �    q  `  N  ;  !    �  �  �  �  q  P  3    �  �  �  �  �  �  �  n  I  $  �  �  o    �  n    �  5  �  t  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        �     	      �  �  �  �  �  �  _  �  �  "  �  .  �  �  �  �          �  �  �  �  Q    �  �  b    �  |  �  t  �  �  �  �  �  z  h  W  F  4  !    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  g  i  e  X  K  O  >  +    �  �  �  q  '  �  v    �  �  �  �  �  r  P  7      �  �  �  ~  C  �  �    �    �  �  w  	�  	�  	�  	�  	e  	$  �  �  H  �  �  C  �  �  s    �  �  �  E  b  ]  W  R  M  H  C  >  8  1  +  $          �  �  �  �  �  �  �  �  j  J  0    �  �  �  n  ?        �  �    &      �  �  �  �  �  �  �  �  s  d  s  �  �  h  P  !  �  �  �  �  �  �  h  N  3    �  �  �  �  �  w  h  X  F  0    �  �  �  �  |  k  Y  H  6  &    	  �  �  �  �  �  n  5  �  �  �  �  �  �  �  �  �  �  �  �  p  W  >  %  	  �  �  �  M          �  �  �  �  �  �  �  u  >  �  �  a    �  c    �  '    �  �  �  �  z  O    �  �  w  :  �  �  e    �  �  M  X  H  :  ,      �  �  �  �  �  k  I  2    �  �  �  a  A  �  �  �  �  �  �  r  U  8  '    �  �  �  v  2  �  I  �  �  .  ]  e  c  k  r  e  W  H  8    �  �  �  ]     �  �    N  �  �  �  �  �  �  �  n  N  +    �  �    N    �  �  �  Z  �  �  }  q  ]  H  2      �  �  �  �  �  y  \  >     �   �  �  w  [  >  "    �  �  �  �  �  �  �  �  �  �  |  u  m  d  F  z  ]  d  t  U  :    �  �  �  A  �  �  D  �  c  �      |  q  d  V  B  .    �  �  �  �  p  J  $  �  �  �  �  �  �  �  �  �  �  �  �  }  _  >    �  �  �  �  R    �  �  �  m  �  �  z  m  _  R  E  7  )      �  �  �  �  �  �  B  �  �  �  �  �  �  �  �  �  �  �  �  n  M  #  �  �  �  I    �  �          �  �  �  T    �  �  �  s  b  D    �  �  {    �  �  �  �  �  f  ;    �  �  t  #  �  �  J    �  q    �  C  1      �  �  �  �  �  o  X  E  3  !       �   �   �   �  �  �  �  �  z  l  ^  P  A  4  )          �  �  J    �  �  �  q  ,  
�  
�  
*  	�  
o  
X  
3  
  	�  	�  	<  �  �  u  (  ~